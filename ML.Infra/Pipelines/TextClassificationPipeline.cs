using System.Numerics.Tensors;
using ML.Infra.Abstractions;
using ML.Infra.Tokenization;

namespace ML.Infra.Pipelines;


public record ClassificationResult<T>(T Choice, float Score, IReadOnlyCollection<float> Logits);
public record TextClassificationOptions<TClassification>(TClassification[] Choices);

public class TextClassificationPipeline<TClassification> : Pipeline<string, ClassificationResult<TClassification>, BatchTokenizedResult, Tensor<float>[]>
{
    private readonly PretrainedTokenizer _tokenizer;
    private readonly IModelExecutor<long, float> _modelExecutor;
    private readonly TextClassificationOptions<TClassification> _pipelineOptions;

    public TextClassificationPipeline(PretrainedTokenizer tokenizer, IModelExecutor<long, float> modelExecutor,
        TextClassificationOptions<TClassification> textClassificationOptions,
        IPipelineBatchExecutor<string, ClassificationResult<TClassification>> executor) : base(executor)
    {
        _tokenizer = tokenizer;
        _modelExecutor = modelExecutor;
        _pipelineOptions = textClassificationOptions;
    }

    protected override BatchTokenizedResult Preprocess(ReadOnlySpan<string> input)
    {
        return _tokenizer.BatchTokenize(input);
    }

    protected override async Task<Tensor<float>[]> RunModel(ReadOnlyMemory<string> input, BatchTokenizedResult tokenizedResult)
    {
        return await _modelExecutor.RunAsync([tokenizedResult.Tokens, tokenizedResult.Mask]);
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
        int argmax = TensorPrimitives.IndexOfMax<float>(probabilities);
        float score = TensorPrimitives.Max<float>(probabilities);
        return new ClassificationResult<TClassification>(_pipelineOptions.Choices[argmax], score, logits.ToArray());
    }
}
