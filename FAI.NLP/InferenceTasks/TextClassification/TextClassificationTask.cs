using System.Numerics.Tensors;
using FAI.Core;
using FAI.Core.Abstractions;
using FAI.Core.ResultTypes;
using FAI.NLP.Configuration;
using FAI.NLP.Tokenization;

namespace FAI.NLP.InferenceTasks.TextClassification;

/// <summary>
/// Represents a pipeline for text classification tasks.
/// </summary>
/// <typeparam name="TClassification">The type of classification labels.</typeparam>
public class TextClassification<TClassification>
    : InferenceSteps<TokenizedText, BatchTokenizedResult, ClassificationResult<TClassification>[], ClassificationResult<TClassification>>
{
    private readonly PretrainedTokenizer _tokenizer;
    private readonly IModelExecutor<long, float> _modelExecutor;
    private readonly TextClassificationOptions<TClassification> _pipelineOptions;

    /// <summary>
    /// Initializes a new instance of the <see cref="TextClassification{TClassification}"/> class.
    /// </summary>
    /// <param name="tokenizer">The pretrained tokenizer used for text tokenization.</param>
    /// <param name="modelExecutor">The model executor used for inference.</param>
    /// <param name="textClassificationOptions">The configuration options for text classification.</param>
    public TextClassification(
        PretrainedTokenizer tokenizer,
        IModelExecutor<long, float> modelExecutor,
        TextClassificationOptions<TClassification> textClassificationOptions)
    {
        _tokenizer = tokenizer;
        _modelExecutor = modelExecutor;
        _pipelineOptions = textClassificationOptions;
    }

    /// <summary>
    /// Preprocesses input tokenized text, converting it into a batch tokenized result.
    /// </summary>
    /// <param name="input">The input tokenized text.</param>
    /// <returns>A batch tokenized result, including tokenization and masks.</returns>
    public override BatchTokenizedResult Preprocess(ReadOnlySpan<TokenizedText> input)
    {
        if (input[0].Tokens is null)
        {
            return _tokenizer.BatchTokenize(new TextView(input));
        }

        (Tensor<long> tokenization, Tensor<long> mask) = _tokenizer.BatchTokensToTensors(new TokensView(input));

        return new BatchTokenizedResult(tokenization, mask);
    }

    /// <summary>
    /// Executes the model inference using the tokenized text and produces classification results.
    /// </summary>
    /// <param name="input">The input tokenized text.</param>
    /// <param name="tokenizedResult">The preprocessed batch tokenized result.</param>
    /// <returns>
    /// A task containing an array of <see cref="ClassificationResult{TClassification}"/> corresponding to each input.
    /// </returns>
    public override async Task<ClassificationResult<TClassification>[]> RunModel(
        ReadOnlyMemory<TokenizedText> input,
        BatchTokenizedResult tokenizedResult)
    {
        var outputs = new ClassificationResult<TClassification>[input.Length];
        await _modelExecutor.RunAsync([tokenizedResult.Tokens, tokenizedResult.Mask], (logits, _) =>
        {
            int indexInBatch = 0;
            foreach (ReadOnlyTensorSpan<float> rowLogits in logits.GetDimensionSpan(0))
            {
                outputs[indexInBatch] = GetClassificationResult(rowLogits.AsSpan());
                indexInBatch++;
            }
        });
        return outputs;
    }

    /// <summary>
    /// Post-processes the model outputs into the final classification results.
    /// </summary>
    /// <param name="inputs">The input tokenized text.</param>
    /// <param name="preprocesses">The preprocessed batch tokenized result.</param>
    /// <param name="modelOutput">The raw classification results from the model.</param>
    /// <param name="outputs">The final classification results to be populated.</param>
    public override void PostProcess(
        ReadOnlySpan<TokenizedText> inputs,
        BatchTokenizedResult preprocesses,
        ClassificationResult<TClassification>[] modelOutput,
        Span<ClassificationResult<TClassification>> outputs)
    {
        modelOutput.AsSpan().CopyTo(outputs);
    }

    /// <summary>
    /// Generates a classification result from the raw logits produced by the model.
    /// </summary>
    /// <param name="logits">The raw logits produced by the model.</param>
    /// <returns>A classification result containing the predicted label and confidence score.</returns>
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