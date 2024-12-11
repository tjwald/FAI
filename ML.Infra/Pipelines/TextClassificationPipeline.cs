using Microsoft.ML.OnnxRuntime;
using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using CommunityToolkit.HighPerformance;
using ML.Infra.Tokenization;
using ML.Infra.Abstractions;
using MemoryExtensions = System.MemoryExtensions;


namespace ML.Infra.Pipelines;


public record ClassificationResult<T>(T Choice, float Score, IReadOnlyCollection<float> Logits);
public record TextClassificationOptions<TClassification>(TClassification[] Choices);

public class TextClassificationPipeline<TClassification> : Pipeline<string, ClassificationResult<TClassification>, BatchTokenizedResult, Tensor<float>[]>
{
    private readonly PretrainedTokenizer _tokenizer;
    private readonly ModelRunner _modelRunner;
    private readonly TextClassificationOptions<TClassification> _pipeLineOptions;

    public TextClassificationPipeline(PretrainedTokenizer tokenizer, ModelRunner modelRunner,
        TextClassificationOptions<TClassification> textClassificationOptions,
        IPipelineBatchExecutor<string, ClassificationResult<TClassification>> executor) : base(executor)
    {
        _tokenizer = tokenizer;
        _modelRunner = modelRunner;
        _pipeLineOptions = textClassificationOptions;
    }

    protected override BatchTokenizedResult Preprocess(ReadOnlySpan<string> input)
    {
        return _tokenizer.BatchTokenize(input);
    }

    protected override async Task<Tensor<float>[]> RunModel(ReadOnlyMemory<string> input, BatchTokenizedResult tokenizedResult)
    {
        return await _modelRunner.RunAsync([tokenizedResult.Tokens, tokenizedResult.Mask]);
    }

    protected override void PostProcess(ReadOnlySpan<string> inputs, BatchTokenizedResult preprocesses, Tensor<float>[] modelResult, Span<ClassificationResult<TClassification>> outputs)
    {
        TensorSpan<float> logits = modelResult[0].AsTensorSpan();

        for (int indexInBatch = 0; indexInBatch < logits.Lengths[0]; indexInBatch++)
        {
            ReadOnlySpan<float> rowLogits = logits.GetRowSpan(indexInBatch);
            outputs[indexInBatch] = GetClassificationResult(rowLogits);
        }
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
        IPipelineBatchExecutor<string, ClassificationResult<TClassification>> executor)
    {
        Task<PretrainedTokenizer> tokenizer = TokenizationUtils.BpeTokenizerFromPretrained(modelDir, tokenizerOptions);
        Task<ModelRunner> modelRunner = ModelRunner.FromPretrained(modelDir, modelRunnerOptions);
        await Task.WhenAll(tokenizer, modelRunner);
        return new TextClassificationPipeline<TClassification>(tokenizer.Result, modelRunner.Result, textClassificationOptions, executor);
    }
}
