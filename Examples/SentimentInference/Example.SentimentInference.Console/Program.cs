using System.Text.Json.Serialization;
using Example.SentimentInference.Model;
using FAI.Extensions.Evaluation;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Parquet;
using Parquet.Data;
using ServiceDefaults;

var builder = Host.CreateApplicationBuilder(args);

builder.ConfigureOpenTelemetry();

const string fileName = "train-00000-of-00001.parquet";

var options = SentimentInferenceOptions.DefaultConfig with { UseGpu = !args.Contains("--cpu", StringComparer.OrdinalIgnoreCase) };

builder.Services.AddDefaultSentimentInference(options);
builder.Services.AddSingleton(new EvaluationPipelineOptions());
builder.Services.AddSingleton<IDataLoader<string, TrainingData, string>, TrainingParquetReader>();
builder.Services.AddSingleton<IEvaluator<TrainingData, bool, EvaluationSummary>, Evaluator>();
builder.Services.AddSingleton<EvaluationPipeline<string, TrainingData, string, bool, EvaluationSummary>>();

var app = builder.Build();

var evaluationPipeline = app.Services.GetRequiredService<EvaluationPipeline<string, TrainingData, string, bool, EvaluationSummary>>();
var logger = app.Services.GetRequiredService<ILogger<Program>>();
var evaluationPipelineResult = await evaluationPipeline.Evaluate(fileName);

logger.LogInformation($"elapsed time: {evaluationPipelineResult.InferenceRuntime.TotalSeconds}s");
logger.LogInformation($"avg time: {evaluationPipelineResult.AveragePerSample.TotalMilliseconds}ms/it");
logger.LogInformation($"Correct predictions: {evaluationPipelineResult.Evaluation}");

internal sealed class TrainingData : IInferenceInputGetter<string>
{
    [JsonPropertyName("sentence")] public string Sentence { get; set; } = null!;
    [JsonPropertyName("label")] public long Label { get; set; }

    public string InferenceInput => Sentence;
}

internal class TrainingParquetReader : IDataLoader<string, TrainingData, string>
{
    public async IAsyncEnumerable<TrainingData> LoadData(string filePath)
    {
        await using Stream fs = File.OpenRead(filePath);
        using ParquetReader reader = await ParquetReader.CreateAsync(fs);
        var sentenceField = reader.Schema.FindDataField("sentence");
        var labelField = reader.Schema.FindDataField("label");

        for (int i = 0; i < reader.RowGroupCount; i++)
        {
            using ParquetRowGroupReader rowGroupReader = reader.OpenRowGroupReader(i);
            DataColumn sentenceColumn = await rowGroupReader.ReadColumnAsync(sentenceField);
            DataColumn labelColumn = await rowGroupReader.ReadColumnAsync(labelField);
            for (int j = 0; j < sentenceColumn.Data.Length; j++)
            {
                var trainingData = new TrainingData { Sentence = (string)sentenceColumn.Data.GetValue(j)!, Label = (long)labelColumn.Data.GetValue(j)! };
                yield return trainingData;
            }
        }
    }
}


internal class Evaluator : IEvaluator<TrainingData, bool, EvaluationSummary>
{
    private readonly ILogger<Evaluator> _logger;

    public Evaluator(ILogger<Evaluator> logger)
    {
        _logger = logger;
    }

    public async Task<EvaluationSummary> Evaluate(IAsyncEnumerable<(TrainingData[], bool[])> inferenceResults)
    {
        int count = 0;
        int correct = 0;
        await foreach (var (inputs, outputs) in inferenceResults)
        {
            count += inputs.Length;
            correct += inputs.Zip(outputs).Count(s => (s.First.Label == 1) == s.Second);
        }

        _logger.LogInformation("Total count: {count}", count);
        _logger.LogInformation("Correct predictions: {correct}", correct);
        _logger.LogInformation("Incorrect predictions: {incorrect}", count - correct);
        _logger.LogInformation("Accuracy: {accuracy}%", correct * 100.0 / count);
        return new EvaluationSummary(count, correct);
    }
}

internal record EvaluationSummary(int SampleSize, int Correct)
{
    public double Accuracy => (double)Correct / SampleSize;

    public override string ToString()
    {
        return $"{Correct}/{SampleSize}={Accuracy:P2}";
    }
}
