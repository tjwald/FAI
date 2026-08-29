using System.Numerics.Tensors;
using Example.TextEmbedding.Model;
using FAI.Core;
using FAI.Extensions.Evaluation;
using Microsoft.Extensions.Logging.Abstractions;
using Parquet;
using Parquet.Data;

namespace Example.TextEmbedding.ConsoleApp;

internal sealed record PerformanceSentence(string Text) : IInferenceInputGetter<string>
{
    public string InferenceInput => Text;
}

internal sealed record StsSentence(int PairIndex, bool IsFirst, string Text, double Score) : IInferenceInputGetter<string>
{
    public string InferenceInput => Text;
}

internal sealed record StsEvaluationSummary(int PairCount, double Pearson, double Spearman);

internal static class StsBenchmark
{
    public static async Task RunAsync(TextEmbeddingInference embeddings, string datasetPath, string performanceDatasetPath)
    {
        _ = await embeddings.BatchPredict(new string[]
        {
            "Warm-up text excluded from the measured benchmark corpus.",
            "A second warm-up sentence initializes batched model execution."
        });

        var performancePipeline = CreatePipeline(
            new PerformanceParquetReader(),
            embeddings,
            new PerformanceEvaluator());
        EvaluationPipelineResult<int> performance = await performancePipeline.Evaluate(performanceDatasetPath);

        System.Console.WriteLine("\nEmbedding throughput results");
        System.Console.WriteLine($"Unique sentences:     {performance.SampleSize:N0}");
        System.Console.WriteLine($"Elapsed:              {performance.InferenceRuntime.TotalSeconds:F2} s");
        System.Console.WriteLine($"Throughput:           {performance.SampleSize / performance.InferenceRuntime.TotalSeconds:F2} sentences/s");

        var qualityPipeline = CreatePipeline(
            new StsParquetReader(),
            embeddings,
            new StsEvaluator());
        EvaluationPipelineResult<StsEvaluationSummary> quality = await qualityPipeline.Evaluate(datasetPath);

        System.Console.WriteLine("\nSTS-B quality validation results");
        System.Console.WriteLine($"Pairs:                {quality.Evaluation.PairCount:N0}");
        System.Console.WriteLine($"Pearson correlation:  {quality.Evaluation.Pearson:F4}");
        System.Console.WriteLine($"Spearman correlation: {quality.Evaluation.Spearman:F4}");
    }

    private static EvaluationPipeline<string, TLoadedInput, string, Tensor<float>, TEvaluationResult>
        CreatePipeline<TLoadedInput, TEvaluationResult>(
            IDataLoader<string, TLoadedInput, string> loader,
            TextEmbeddingInference embeddings,
            IEvaluator<TLoadedInput, Tensor<float>, TEvaluationResult> evaluator)
        where TLoadedInput : IInferenceInputGetter<string>
    {
        return new(
            loader,
            embeddings,
            evaluator,
            NullLogger<EvaluationPipeline<string, TLoadedInput, string, Tensor<float>, TEvaluationResult>>.Instance,
            new EvaluationPipelineOptions());
    }
}

internal sealed class PerformanceParquetReader : IDataLoader<string, PerformanceSentence, string>
{
    private const int SentenceCount = 100_000;

    public async IAsyncEnumerable<PerformanceSentence> LoadData(string datasetPath)
    {
        var sentences = new HashSet<string>(StringComparer.Ordinal);
        await using Stream stream = File.OpenRead(datasetPath);
        using ParquetReader reader = await ParquetReader.CreateAsync(stream);
        var textField = reader.Schema.FindDataField("text");

        for (int rowGroupIndex = 0; rowGroupIndex < reader.RowGroupCount && sentences.Count < SentenceCount; rowGroupIndex++)
        {
            using ParquetRowGroupReader rowGroup = reader.OpenRowGroupReader(rowGroupIndex);
            DataColumn texts = await rowGroup.ReadColumnAsync(textField);
            foreach (object? value in texts.Data)
            {
                if (value is string text && !string.IsNullOrWhiteSpace(text) && sentences.Add(text))
                {
                    yield return new(text);
                    if (sentences.Count == SentenceCount)
                    {
                        yield break;
                    }
                }
            }
        }

        throw new InvalidDataException(
            $"Performance dataset contains only {sentences.Count:N0} distinct sentences; {SentenceCount:N0} are required.");
    }
}

internal sealed class PerformanceEvaluator : IEvaluator<PerformanceSentence, Tensor<float>, int>
{
    public async Task<int> Evaluate(IAsyncEnumerable<(PerformanceSentence[], Tensor<float>)> inferenceResults)
    {
        int sentenceCount = 0;
        await foreach (var (inputs, outputs) in inferenceResults)
        {
            if (outputs.Rank != 2 || outputs.Lengths[0] != inputs.Length)
            {
                throw new InvalidDataException("Embedding output rows must match the evaluated input count.");
            }

            sentenceCount += inputs.Length;
        }

        return sentenceCount;
    }
}

internal sealed class StsParquetReader : IDataLoader<string, StsSentence, string>
{
    public async IAsyncEnumerable<StsSentence> LoadData(string datasetPath)
    {
        await using Stream stream = File.OpenRead(datasetPath);
        using ParquetReader reader = await ParquetReader.CreateAsync(stream);
        var sentence1Field = reader.Schema.FindDataField("sentence1");
        var sentence2Field = reader.Schema.FindDataField("sentence2");
        var scoreField = reader.Schema.FindDataField("score");
        int pairIndex = 0;

        for (int rowGroupIndex = 0; rowGroupIndex < reader.RowGroupCount; rowGroupIndex++)
        {
            using ParquetRowGroupReader rowGroup = reader.OpenRowGroupReader(rowGroupIndex);
            DataColumn sentence1 = await rowGroup.ReadColumnAsync(sentence1Field);
            DataColumn sentence2 = await rowGroup.ReadColumnAsync(sentence2Field);
            DataColumn scores = await rowGroup.ReadColumnAsync(scoreField);
            for (int rowIndex = 0; rowIndex < sentence1.Data.Length; rowIndex++)
            {
                double score = Convert.ToDouble(scores.Data.GetValue(rowIndex));
                yield return new(pairIndex, true, (string)sentence1.Data.GetValue(rowIndex)!, score);
                yield return new(pairIndex, false, (string)sentence2.Data.GetValue(rowIndex)!, score);
                pairIndex++;
            }
        }
    }
}

internal sealed class StsEvaluator : IEvaluator<StsSentence, Tensor<float>, StsEvaluationSummary>
{
    public async Task<StsEvaluationSummary> Evaluate(IAsyncEnumerable<(StsSentence[], Tensor<float>)> inferenceResults)
    {
        var expectedScores = new List<double>();
        var predictedScores = new List<double>();

        await foreach (var (inputs, outputs) in inferenceResults)
        {
            if (outputs.Rank != 2 || outputs.Lengths[0] != inputs.Length || inputs.Length % 2 != 0)
            {
                throw new InvalidDataException("STS embedding output must contain two rows per sentence pair.");
            }

            int dimensions = checked((int)outputs.Lengths[1]);
            ReadOnlySpan<float> values = outputs.AsMemory().Span;
            for (int index = 0; index < inputs.Length; index += 2)
            {
                StsSentence first = inputs[index];
                StsSentence second = inputs[index + 1];
                if (!first.IsFirst || second.IsFirst || first.PairIndex != second.PairIndex || first.Score != second.Score)
                {
                    throw new InvalidDataException("STS sentence pairs were not preserved during inference.");
                }

                expectedScores.Add(first.Score);
                predictedScores.Add(TensorPrimitives.Dot(
                    values.Slice(index * dimensions, dimensions),
                    values.Slice((index + 1) * dimensions, dimensions)));
            }
        }

        double[] expected = expectedScores.ToArray();
        double[] predicted = predictedScores.ToArray();
        return new(expected.Length, Correlation(expected, predicted), Correlation(Rank(expected), Rank(predicted)));
    }

    private static double Correlation(ReadOnlySpan<double> left, ReadOnlySpan<double> right)
    {
        double leftMean = 0;
        double rightMean = 0;
        for (int index = 0; index < left.Length; index++)
        {
            leftMean += left[index];
            rightMean += right[index];
        }

        leftMean /= left.Length;
        rightMean /= right.Length;

        double covariance = 0;
        double leftVariance = 0;
        double rightVariance = 0;
        for (int index = 0; index < left.Length; index++)
        {
            double leftDelta = left[index] - leftMean;
            double rightDelta = right[index] - rightMean;
            covariance += leftDelta * rightDelta;
            leftVariance += leftDelta * leftDelta;
            rightVariance += rightDelta * rightDelta;
        }

        return covariance / Math.Sqrt(leftVariance * rightVariance);
    }

    private static double[] Rank(double[] values)
    {
        int[] order = Enumerable.Range(0, values.Length).OrderBy(index => values[index]).ToArray();
        var ranks = new double[values.Length];
        int start = 0;
        while (start < order.Length)
        {
            int end = start + 1;
            while (end < order.Length && values[order[end]].Equals(values[order[start]]))
            {
                end++;
            }

            double averageRank = (start + end - 1) / 2.0;
            for (int index = start; index < end; index++)
            {
                ranks[order[index]] = averageRank;
            }

            start = end;
        }

        return ranks;
    }
}
