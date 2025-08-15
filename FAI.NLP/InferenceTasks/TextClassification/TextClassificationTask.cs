using System.Numerics.Tensors;
using FAI.Core;
using FAI.Core.Abstractions;
using FAI.Core.Configurations.InferenceTasks;
using FAI.Core.InferenceTasks.Classification;
using FAI.Core.ResultTypes;
using FAI.NLP.Tokenization;

namespace FAI.NLP.InferenceTasks.TextClassification;

/// <summary>
/// Represents a pipeline for text classification tasks.
/// </summary>
/// <typeparam name="TClassification">The type of classification labels.</typeparam>
public class TextClassification<TClassification>
    : InferenceSteps<TokenizedText, BatchTokenizedResult, ClassificationResult<TClassification, float>[], ClassificationResult<TClassification, float>>
{
    private readonly PretrainedTokenizer _tokenizer;
    private readonly IModelExecutor<long, float> _modelExecutor;
    private readonly ClassificationOptions<TClassification> _pipelineOptions;

    /// <summary>
    /// Initializes a new instance of the <see cref="TextClassification{TClassification}"/> class.
    /// </summary>
    /// <param name="tokenizer">The pretrained tokenizer used for text tokenization.</param>
    /// <param name="modelExecutor">The model executor used for inference.</param>
    /// <param name="textClassificationOptions">The configuration options for text classification.</param>
    public TextClassification(
        PretrainedTokenizer tokenizer,
        IModelExecutor<long, float> modelExecutor,
        ClassificationOptions<TClassification> textClassificationOptions)
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
    public override async Task<ClassificationResult<TClassification, float>[]> RunModel(
        ReadOnlyMemory<TokenizedText> input,
        BatchTokenizedResult tokenizedResult)
    {
        var outputs = new ClassificationResult<TClassification, float>[input.Length];
        await _modelExecutor.RunAsync([tokenizedResult.Tokens, tokenizedResult.Mask], (logits, _) =>
        {
            int indexInBatch = 0;
            foreach (ReadOnlyTensorSpan<float> rowLogits in logits.GetDimensionSpan(0))
            {
                outputs[indexInBatch] = _pipelineOptions.GetClassificationResult(rowLogits.AsSpan());
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
        ClassificationResult<TClassification, float>[] modelOutput,
        Span<ClassificationResult<TClassification, float>> outputs)
    {
        modelOutput.AsSpan().CopyTo(outputs);
    }
}