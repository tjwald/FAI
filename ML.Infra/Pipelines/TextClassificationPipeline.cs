using Microsoft.ML.OnnxRuntime;
using System.Numerics.Tensors;
using CommunityToolkit.HighPerformance;
using ML.Infra.Tokenization;
using ML.Infra.Abstractions;


namespace ML.Infra.Pipelines;


public record ClassificationResult<T>(T Choice, float Score, IReadOnlyCollection<float> Logits);
public record TextClassificationOptions<TClassification>(TClassification[] Choices);

public class TextClassificationPipeline<TClassification> : Pipeline<string, ClassificationResult<TClassification>, BatchTokenizedResult, IDisposableReadOnlyCollection<OrtValue>>
{
    private readonly PretrainedTokenizer _tokenizer;
    private readonly ModelRunner _modelRunner;
    private readonly TextClassificationOptions<TClassification> _pipeLineOptions;

    public TextClassificationPipeline(PretrainedTokenizer tokenizer, ModelRunner modelRunner,
        TextClassificationOptions<TClassification> textClassificationOptions,
        IPipelineBatchExecutor<string,ClassificationResult<TClassification>,BatchTokenizedResult,IDisposableReadOnlyCollection<OrtValue>> executor) : base(executor)
    {
        _tokenizer = tokenizer;
        _modelRunner = modelRunner;
        _pipeLineOptions = textClassificationOptions;
    }

    protected override BatchTokenizedResult Preprocess(ReadOnlySpan<string> input)
    {
        return _tokenizer.BatchTokenize(input);
    }

    protected override async Task<IDisposableReadOnlyCollection<OrtValue>> RunModel(ReadOnlyMemory<string> input, BatchTokenizedResult tokenizedResult)
    {
        Span<Memory<long>> modelInputs = [tokenizedResult.Tokens.AsMemory(), tokenizedResult.Mask.AsMemory()];
        return await _modelRunner.RunAsync(modelInputs.ToOrtValues([tokenizedResult.BatchSize, tokenizedResult.MaxTokenCount]));
    }

    protected override void PostProcess(ReadOnlySpan<string> inputs, BatchTokenizedResult preprocesses, IDisposableReadOnlyCollection<OrtValue> modelResult, Span<ClassificationResult<TClassification>> outputs)
    {
        var logits = modelResult[0].GetTensorDataAsSpan<float>().AsSpan2D(preprocesses.BatchSize, _pipeLineOptions.Choices.Length);

        for (int indexInBatch = 0; indexInBatch < logits.Height; indexInBatch++)
        {
            ReadOnlySpan<float> rowLogits = logits.GetRowSpan(indexInBatch);
            outputs[indexInBatch] = GetClassificationResult(rowLogits);
        }

        modelResult.Dispose();
    }

    private ClassificationResult<TClassification> GetClassificationResult(ReadOnlySpan<float> logits)
    {
        Span<float> probabilities = stackalloc float[logits.Length];
        TensorPrimitives.SoftMax(logits, probabilities);
        int argmax = TensorPrimitives.IndexOfMax(probabilities);
        float score = TensorPrimitives.Max(probabilities);
        return new ClassificationResult<TClassification>(_pipeLineOptions.Choices[argmax], score, logits.ToArray());
    }

    public static async Task<TextClassificationPipeline<TClassification>> FromPretrained(string modelDir,
        TextClassificationOptions<TClassification> textClassificationOptions, PretrainedTokenizerOptions tokenizerOptions,
        OnnxModelRunerOptions modelRunnerOptions,
        IPipelineBatchExecutor<string,ClassificationResult<TClassification>,BatchTokenizedResult,IDisposableReadOnlyCollection<OrtValue>> executor)
    {
        Task<PretrainedTokenizer> tokenizer = TokenizationUtils.BpeTokenizerFromPretrained(modelDir, tokenizerOptions);
        Task<ModelRunner> modelRunner = ModelRunner.FromPretrained(modelDir, modelRunnerOptions);
        await Task.WhenAll(tokenizer, modelRunner);
        return new TextClassificationPipeline<TClassification>(tokenizer.Result, modelRunner.Result, textClassificationOptions, executor);
    }
}
