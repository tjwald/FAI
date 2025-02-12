using System.Numerics.Tensors;
using ML.Infra.Abstractions;
using ML.Infra.Configurations.Pipelines;
using ML.Infra.ResultTypes;
using ML.Infra.Tokenization;

namespace ML.Infra.Pipelines;

public class TextClassificationPipeline<TClassification>: Pipeline<TokenizedText, ClassificationResult<TClassification>, BatchTokenizedResult, Tensor<float>[]>
{
    private readonly PretrainedTokenizer _tokenizer;
    private readonly IModelExecutor<long, float> _modelExecutor;
    private readonly TextClassificationOptions<TClassification> _pipelineOptions;

    public TextClassificationPipeline(PretrainedTokenizer tokenizer, IModelExecutor<long, float> modelExecutor,
        TextClassificationOptions<TClassification> textClassificationOptions,
        IPipelineBatchExecutor<TokenizedText, ClassificationResult<TClassification>> executor) : base(executor)
    {
        _tokenizer = tokenizer;
        _modelExecutor = modelExecutor;
        _pipelineOptions = textClassificationOptions;
    }

    public override BatchTokenizedResult Preprocess(ReadOnlySpan<TokenizedText> input)
    {
        if (input[0].Tokens is null)
        {
            return _tokenizer.BatchTokenize(new TextView(input));
        }

        (Tensor<long> tokenization, Tensor<long> mask) = _tokenizer.BatchTokensToTensors(new TokensView(input));
        
        return new BatchTokenizedResult(tokenization, mask);
    }

    public override async Task<Tensor<float>[]> RunModel(ReadOnlyMemory<TokenizedText> input, BatchTokenizedResult tokenizedResult)
    {
        return await _modelExecutor.RunAsync([tokenizedResult.Tokens, tokenizedResult.Mask]);
    }

    public override void PostProcess(ReadOnlySpan<TokenizedText> inputs, BatchTokenizedResult preprocesses, Tensor<float>[] modelResult, Span<ClassificationResult<TClassification>> outputs)
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