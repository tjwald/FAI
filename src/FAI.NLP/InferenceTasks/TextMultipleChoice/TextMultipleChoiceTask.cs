using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using FAI.Core;
using FAI.Core.Abstractions;
using FAI.Core.ResultTypes;
using FAI.NLP.Configuration;
using FAI.NLP.Tokenization;

namespace FAI.NLP.InferenceTasks.TextMultipleChoice;

/// <summary>
/// Represents a pipeline for multiple-choice text classification tasks.
/// </summary>
public class TextMultipleChoiceTask : InferenceSteps<TextMultipleChoiceInput, BatchTokenizedResult, ChoiceResult<TokenizedText>[], ChoiceResult<TokenizedText>>
{
    private readonly PretrainedTokenizer _tokenizer;
    private readonly TextMultipleChoiceOptions _options;
    private readonly IModelExecutor<long, float> _modelExecutor;

    /// <summary>
    /// Initializes a new instance of the <see cref="TextMultipleChoiceTask"/> class.
    /// </summary>
    /// <param name="tokenizer">The pretrained tokenizer used for tokenizing inputs and choices.</param>
    /// <param name="modelExecutor">The model executor used for inference.</param>
    /// <param name="options">The options for configuring multiple-choice tasks.</param>
    public TextMultipleChoiceTask(PretrainedTokenizer tokenizer, IModelExecutor<long, float> modelExecutor, TextMultipleChoiceOptions options)
    {
        _tokenizer = tokenizer;
        _options = options;
        _modelExecutor = modelExecutor;
    }

    /// <summary>
    /// Preprocesses the input into a batch tokenized result, preparing tensors for inference.
    /// </summary>
    /// <param name="input">The input containing context and multiple-choice options.</param>
    /// <returns>A batch tokenized result with token and mask tensors.</returns>
    public override BatchTokenizedResult Preprocess(ReadOnlySpan<TextMultipleChoiceInput> input)
    {
        (List<List<int>?> tokensList, int maxChoiceCount, int maxTokenCount) =
            input[0].IsTokenized ? FlattenTokensWithPadding(input) : FlattenBatchTokenize(input);

        Tensor<long> tokenTensor = Tensor.CreateFromShape<long>([input.Length * maxChoiceCount, maxTokenCount]);
        Tensor<long> maskTensor = Tensor.CreateFromShape<long>([input.Length * maxChoiceCount, maxTokenCount]);

        var tokenTensorSpan = tokenTensor.GetDimensionSpan(0);
        var maskTensorSpan = maskTensor.GetDimensionSpan(0);

        int outputRow = 0;
        foreach (List<int>? tokens in tokensList)
        {
            if (tokens is null)
            {
                // skip writing tokens and mask since mask = 0 by default, and tokens don't matter when mask = 0
                outputRow += _options.MaxChoices - outputRow % _options.MaxChoices;
                continue;
            }

            var tokenRow = tokenTensorSpan[outputRow].AsSpan();
            var maskRow = maskTensorSpan[outputRow].AsSpan();

            TensorPrimitives.ConvertChecked<int, long>(CollectionsMarshal.AsSpan(tokens), tokenRow);

            maskRow[..tokens.Count].Fill(1);
            outputRow++;
        }

        Span<nint> shape = [input.Length, maxChoiceCount, maxTokenCount];
        return new BatchTokenizedResult(tokenTensor.Reshape(shape), maskTensor.Reshape(shape));
    }

    private (List<List<int>?> tokensList, int maxChoiceCount, int maxTokenCount) FlattenBatchTokenize(ReadOnlySpan<TextMultipleChoiceInput> input)
    {
        List<List<int>?> tokens = [];
        int maxTokenCount = 0;
        int maxChoiceCount = 0;
        foreach ((string context, TokenizedText[] choices) in input)
        {
            if (choices.Length > _options.MaxChoices)
            {
                throw new InvalidOperationException($"Too many choices for text: {context}");
            }

            maxChoiceCount = Math.Max(maxChoiceCount, choices.Length);
            foreach (var choice in choices)
            {
                List<int> tokenizedText = _tokenizer.Tokenize(context, choice.Text);
                choice.Tokens = tokenizedText;
                maxTokenCount = Math.Max(maxTokenCount, tokenizedText.Count);
                tokens.Add(tokenizedText);
            }

            if (_options.MaxChoices > choices.Length)
            {
                tokens.Add(null);
            }
        }

        return (tokens, maxChoiceCount, maxTokenCount);
    }

    private (List<List<int>?> tokensList, int maxChoiceCount, int maxTokenCount) FlattenTokensWithPadding(ReadOnlySpan<TextMultipleChoiceInput> input)
    {
        List<List<int>?> tokens = [];
        int maxTokenCount = 0;
        int maxChoiceCount = 0;
        foreach (var t in input)
        {
            TokenizedText[] choices = t.Choices;
            maxChoiceCount = Math.Max(maxChoiceCount, choices.Length);
            foreach (var choice in choices)
            {
                maxTokenCount = Math.Max(maxTokenCount, choice.TokenCount);
                tokens.Add(choice.Tokens!);
            }

            if (choices.Length % _options.MaxChoices > 0)
            {
                tokens.Add(null);
            }
        }

        return (tokens, maxChoiceCount, maxTokenCount);
    }

    /// <summary>
    /// Runs the model inference to produce classification results for multiple-choice tasks.
    /// </summary>
    /// <param name="input">The input containing context and choices.</param>
    /// <param name="tokenizedResult">The preprocessed batch tokenized result.</param>
    /// <returns>
    /// A task containing an array of <see cref="ChoiceResult{TokenizedText}"/> representing the classification results.
    /// </returns>
    public override async Task<ChoiceResult<TokenizedText>[]> RunModel(ReadOnlyMemory<TextMultipleChoiceInput> input, BatchTokenizedResult tokenizedResult)
    {
        var outputs = new ChoiceResult<TokenizedText>[input.Length];

        await _modelExecutor.RunAsync([tokenizedResult.Tokens, tokenizedResult.Mask], (logits, _) =>
        {
            int indexInBatch = 0;
            foreach (ReadOnlyTensorSpan<float> rowLogits in logits.GetDimensionSpan(0))
            {
                outputs[indexInBatch] = GetMultipleChoiceResult(input.Span[indexInBatch], rowLogits.AsSpan());
                indexInBatch++;
            }
        });
        return outputs;
    }

    /// <summary>
    /// Post-processes the model outputs into the final classification results.
    /// </summary>
    /// <param name="inputs">The original multiple-choice inputs.</param>
    /// <param name="preprocesses">The preprocessed batch tokenized result.</param>
    /// <param name="modelOutput">The raw classification results from the model.</param>
    /// <param name="outputs">The final choice results to be populated.</param>
    public override void PostProcess(ReadOnlySpan<TextMultipleChoiceInput> inputs, BatchTokenizedResult preprocesses, ChoiceResult<TokenizedText>[] modelOutput,
        Span<ChoiceResult<TokenizedText>> outputs)
    {
        modelOutput.AsSpan().CopyTo(outputs);
    }

    /// <summary>
    /// Generates a multiple-choice result from the raw logits produced by the model.
    /// </summary>
    /// <param name="input">The input containing context and choices.</param>
    /// <param name="logits">The raw logits produced by the model.</param>
    /// <returns>A choice result containing the selected choice, index, and confidence score.</returns>
    private ChoiceResult<TokenizedText> GetMultipleChoiceResult(TextMultipleChoiceInput input, ReadOnlySpan<float> logits)
    {
        Span<float> probabilities = stackalloc float[logits.Length];
        TensorPrimitives.SoftMax(logits, probabilities);
        int argmax = TensorPrimitives.IndexOfMax<float>(probabilities);
        float score = TensorPrimitives.Max<float>(probabilities);

        float[]? logitsArray = _options.StoreLogits ? logits.ToArray() : null;

        return new ChoiceResult<TokenizedText>(input.Choices[argmax], argmax, score, logitsArray);
    }
}
