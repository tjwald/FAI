using Microsoft.ML.OnnxRuntime;
using System.Numerics.Tensors;
using CommunityToolkit.HighPerformance;
using ML.Infra.Tokenization;
using ML.Infra.Abstractions;


namespace ML.Infra.Pipelines;

public record struct BatchTokenizedResultView(Memory2D<long> Tokens, Memory2D<long> Mask)
{
    public int BatchSize { get => Tokens.Height; }
    public int MaxTokenSize { get => Tokens.Width; }
}


public record ClassificationResult<T>(T Choice, float Score, IReadOnlyCollection<float> Logits);
public record TextClassificationOptions<T>(T[] Chioces, int OptimalBatchSize);

public class TextClassificationPipeline<T> : Pipeline<string, ClassificationResult<T>, BatchTokenizedResultView, BatchTokenizedResult, IDisposableReadOnlyCollection<OrtValue>>
{
    private readonly PretrainedTokenizer _tokenizer;
    private readonly ModelRunner _modelRunner;
    private readonly TextClassificationOptions<T> _pipeLineOptions;

    public TextClassificationPipeline(PretrainedTokenizer tokenizer, ModelRunner modelRunner, TextClassificationOptions<T> textClassificationOptions) : base(textClassificationOptions.OptimalBatchSize)
    {
        _tokenizer = tokenizer;
        _modelRunner = modelRunner;
        _pipeLineOptions = textClassificationOptions;
    }

    protected override ValueTask<BatchTokenizedResult> Preprocess(ReadOnlyMemory<string> input)
    {
        return ValueTask.FromResult(_tokenizer.BatchTokenize(input));
    }

    protected override IEnumerable<(int, BatchTokenizedResultView)> BatchPreprocesses(BatchTokenizedResult preprocesses, int optimalBatchSize)
    {
        Memory2D<long> tokenMemory = preprocesses.Tokens.AsMemory2D();
        var maskMemory = preprocesses.Mask.AsMemory2D();

        int i = 0;
        while (i < preprocesses.BatchSize)
        {
            int endRow = Math.Min(i + optimalBatchSize, preprocesses.BatchSize - 1) + 1;

            yield return (endRow - i, new BatchTokenizedResultView(tokenMemory[i..endRow, ..], maskMemory[i..endRow, ..]));

            i = endRow;
        }
    }

    protected override async Task<IDisposableReadOnlyCollection<OrtValue>> RunModel(ReadOnlyMemory<string> input, BatchTokenizedResultView tokenizedResult)
    {
        tokenizedResult.Tokens.TryGetMemory(out var inputIds);
        tokenizedResult.Mask.TryGetMemory(out var attenstionMask);

        Span<Memory<long>> modelInputs = [inputIds, attenstionMask];
        return await _modelRunner.RunAsync(modelInputs.ToOrtValues([tokenizedResult.BatchSize, tokenizedResult.MaxTokenSize]));
    }

    protected override void PostProcess(ReadOnlySpan<string> inputs, BatchTokenizedResultView preprocesses, IDisposableReadOnlyCollection<OrtValue> modelResult, Span<ClassificationResult<T>> outputs)
    {
        var logits = modelResult[0].GetTensorDataAsSpan<float>().AsSpan2D(preprocesses.BatchSize, _pipeLineOptions.Chioces.Length);

        for (int indexInBatch = 0; indexInBatch < logits.Height; indexInBatch++)
        {
            var rowLogits = logits.GetRowSpan(indexInBatch);
            outputs[indexInBatch] = GetClassificationResult(rowLogits);
        }

        modelResult.Dispose();
    }

    private ClassificationResult<T> GetClassificationResult(ReadOnlySpan<float> logits)
    {
        Span<float> probabilities = stackalloc float[logits.Length];
        TensorPrimitives.SoftMax(logits, probabilities);
        var argmax = TensorPrimitives.IndexOfMax(probabilities);
        var score = TensorPrimitives.Max(probabilities);
        return new ClassificationResult<T>(_pipeLineOptions.Chioces[argmax], score, logits.ToArray());
    }

    public static async Task<TextClassificationPipeline<T>> FromPretrained(string modelDir, TextClassificationOptions<T> textClassificationOptions, PretrainedTokenizerOptions tokenizerOptions, OnnxModelRunerOptions modelRunnerOptions)
    {
        var tokenizer = TokenizationUtils.BPETokenizerFromPretrained(modelDir, tokenizerOptions);
        var modelRunner = ModelRunner.FromPretrained(modelDir, modelRunnerOptions);
        await Task.WhenAll(tokenizer, modelRunner);
        return new TextClassificationPipeline<T>(tokenizer.Result, modelRunner.Result, textClassificationOptions);
    }
}
