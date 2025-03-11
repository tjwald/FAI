using System.Numerics.Tensors;
using ML.Infra;
using ML.Infra.Abstractions;
using ML.Infra.ResultTypes;
using ML.NLP.Configuration;
using ML.NLP.Tokenization;

namespace ML.NLP.InferenceTasks;

public class TextClassification<TClassification> 
    : InferenceSteps<TokenizedText, BatchTokenizedResult, ClassificationResult<TClassification>[], ClassificationResult<TClassification>>
{
    private readonly PretrainedTokenizer _tokenizer;
    private readonly IModelExecutor<long, float> _modelExecutor;
    private readonly TextClassificationOptions<TClassification> _pipelineOptions;

    public TextClassification(
        PretrainedTokenizer tokenizer,
        IModelExecutor<long, float> modelExecutor,
        TextClassificationOptions<TClassification> textClassificationOptions)
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

    public override async Task<ClassificationResult<TClassification>[]> RunModel(ReadOnlyMemory<TokenizedText> input, BatchTokenizedResult tokenizedResult)
    {
        var outputs = new ClassificationResult<TClassification>[input.Length];
        await _modelExecutor.RunAsync([tokenizedResult.Tokens, tokenizedResult.Mask], (logits, _) =>
        {
            for (int indexInBatch = 0; indexInBatch < logits.Lengths[0]; indexInBatch++)
            {
                ReadOnlySpan<float> rowLogits = logits.GetRowSpan(indexInBatch);
                outputs[indexInBatch] = GetClassificationResult(rowLogits);
            }
        });
        return outputs;
    }

    public override void PostProcess(ReadOnlySpan<TokenizedText> inputs, BatchTokenizedResult preprocesses, ClassificationResult<TClassification>[] modelOutput,
        Span<ClassificationResult<TClassification>> outputs)
    {
        modelOutput.AsSpan().CopyTo(outputs);
    }
    
    private ClassificationResult<TClassification> GetClassificationResult(ReadOnlySpan<float> logits)
    {
        Span<float> probabilities = stackalloc float[logits.Length];
        TensorPrimitives.SoftMax(logits, probabilities);
        int argmax = TensorPrimitives.IndexOfMax<float>(probabilities);
        float score = TensorPrimitives.Max<float>(probabilities);

        float[]? logitsArray = _pipelineOptions.StoreLogits ? logits.ToArray() : null;
        
        return new ClassificationResult<TClassification>(_pipelineOptions.Choices[argmax], score, logitsArray);
    }
}