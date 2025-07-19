using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using FAI.Core;
using FAI.NLP.Configuration;
using Microsoft.ML.Tokenizers;

namespace FAI.NLP.Tokenization;

/// <summary>
/// Represents a batch tokenized result containing token and mask tensors.
/// </summary>
/// <param name="Tokens">The tensor representing tokenized input sequences.</param>
/// <param name="Mask">
/// The tensor representing the attention mask, indicating which tokens should be processed.
/// </param>
public readonly record struct BatchTokenizedResult(Tensor<long> Tokens, Tensor<long> Mask)
{
    /// <summary>
    /// Gets the batch size, determined by the first dimension of the token tensor.
    /// </summary>
    public int BatchSize => (int)Tokens.Lengths[0];

    /// <summary>
    /// Gets the maximum token count per sequence, determined by the second dimension of the token tensor.
    /// </summary>
    public int MaxTokenCount => (int)Tokens.Lengths[1];
}

/// <summary>
/// Represents a pretrained tokenizer used for tokenizing text inputs and managing token-related transformations.
/// Wraps a <see cref="Microsoft.ML.Tokenizers.Tokenizer"/> and adds batch functionality. 
/// </summary>
public sealed class PretrainedTokenizer
{
    private readonly Tokenizer _tokenizer;
    private readonly PretrainedTokenizerOptions _tokenizerOptions;

    public PretrainedTokenizer(Tokenizer tokenizer, PretrainedTokenizerOptions tokenizerOptions)
    {
        _tokenizer = tokenizer;
        _tokenizerOptions = tokenizerOptions;
    }

    public List<int> Tokenize(string text)
    {
        return (List<int>)_tokenizer.EncodeToIds(text, _tokenizerOptions.MaxTokenLength, out _, out _);
    }

    public List<int> Tokenize(string context, string text)
    {
        if (_tokenizer is not BertTokenizer tokenizer)
        {
            throw new InvalidOperationException("Tokenize with context is not supported on Non-Bert Tokenizer");
        }

        IReadOnlyList<int> tokenizedContext = tokenizer.EncodeToIds(context, addSpecialTokens: false);
        IReadOnlyList<int> tokenizedText = tokenizer.EncodeToIds(text, addSpecialTokens: false);
        (IEnumerable<int> tokenizedContextEnumerable, IEnumerable<int> tokenizedTextEnumerable) = TruncateTokens(tokenizedContext, tokenizedText);

        return (List<int>)tokenizer.BuildInputsWithSpecialTokens(tokenizedContextEnumerable, tokenizedTextEnumerable);
    }

    public Span<int> Tokenize(string context, string text, Span<int> output)
    {
        if (_tokenizer is not BertTokenizer tokenizer)
        {
            throw new InvalidOperationException("Tokenize with context is not supported on Non-Bert Tokenizer");
        }

        IReadOnlyList<int> tokenizedContext = tokenizer.EncodeToIds(context, addSpecialTokens: false);
        IReadOnlyList<int> tokenizedText = tokenizer.EncodeToIds(text, addSpecialTokens: false);

        (IEnumerable<int> tokenizedContextEnumerable, IEnumerable<int> tokenizedTextEnumerable) = TruncateTokens(tokenizedContext, tokenizedText);
        tokenizer.BuildInputsWithSpecialTokens(tokenizedContextEnumerable, output, out int valuesWritten, tokenizedTextEnumerable);
        return output[..valuesWritten];
    }

    private (IEnumerable<int> tokenizedContextEnumerable, IEnumerable<int> tokenizedTextEnumerable) TruncateTokens(IReadOnlyList<int> tokenizedContext,
        IReadOnlyList<int> tokenizedText)
    {
        IEnumerable<int> tokenizedContextEnumerable = tokenizedContext;
        IEnumerable<int> tokenizedTextEnumerable = tokenizedText;
        int truncationLength = tokenizedContext.Count + tokenizedText.Count - _tokenizerOptions.MaxTokenLength;
        if (truncationLength <= 0) return (tokenizedContextEnumerable, tokenizedTextEnumerable);

        switch (_tokenizerOptions.TruncationOption)
        {
            case TruncationOption.Longest when tokenizedContext.Count > tokenizedText.Count:
            case TruncationOption.Context:
                tokenizedContextEnumerable = tokenizedContextEnumerable.SkipLast(truncationLength);
                break;
            case TruncationOption.Longest:
            case TruncationOption.Text:
                tokenizedTextEnumerable = tokenizedTextEnumerable.Take(tokenizedContext.Count - truncationLength);
                break;
            default:
                throw new ArgumentOutOfRangeException($"Tokenization Option: {_tokenizerOptions.TruncationOption} Not supported");
        }

        return (tokenizedContextEnumerable, tokenizedTextEnumerable);
    }

    public BatchTokenizedResult BatchTokenize(TextView inputs)
    {
        int maxTokenSize = 0;
        Span<List<int>> tokenizedInputs = new List<int>[inputs.Count];
        for (int i = 0; i < tokenizedInputs.Length; i++)
        {
            List<int> tokenizedInput = Tokenize(inputs[i]);
            tokenizedInputs[i] = tokenizedInput;
            inputs.SetTokens(i, tokenizedInput);
            if (tokenizedInput.Count > maxTokenSize)
            {
                maxTokenSize = tokenizedInput.Count;
            }
        }

        return BatchTokensToTensors(tokenizedInputs, _tokenizerOptions, maxTokenSize);
    }

    public BatchTokenizedResult BatchTokenize(ReadOnlySpan<string> inputs)
    {
        int maxTokenSize = 0;
        Span<List<int>> tokenizedInputs = new List<int>[inputs.Length];
        for (int i = 0; i < inputs.Length; i++)
        {
            var tokenizedInput = (List<int>)_tokenizer.EncodeToIds(inputs[i], _tokenizerOptions.MaxTokenLength, out _, out _);
            tokenizedInputs[i] = tokenizedInput;
            if (tokenizedInput.Count > maxTokenSize)
            {
                maxTokenSize = tokenizedInput.Count;
            }
        }

        return BatchTokensToTensors(tokenizedInputs, _tokenizerOptions, maxTokenSize);
    }

    public BatchTokenizedResult BatchTokensToTensors(ReadOnlySpan<List<int>> inputs, int maxTokenSize)
    {
        return BatchTokensToTensors(inputs, _tokenizerOptions, maxTokenSize);
    }

    private static BatchTokenizedResult BatchTokensToTensors(ReadOnlySpan<List<int>> inputs, PretrainedTokenizerOptions tokenizerOptions, int maxTokenSize)
    {
        int batchSize = inputs.Length;
        var result = CreateTokenAndMaskTensorsFromShape(batchSize, maxTokenSize);

        TensorDimensionSpan<long> tokenizationSpan = result.Tokens.GetDimensionSpan(0);
        TensorDimensionSpan<long> maskSpan = result.Mask.GetDimensionSpan(0);
        for (int i = 0; i < batchSize; i++)
        {
            TokenizeRow(tokenizerOptions, inputs[i], tokenizationSpan, maskSpan, i);
        }

        return result;
    }

    public BatchTokenizedResult BatchTokensToTensors(TokensView inputs)
    {
        return BatchTokensToTensors(inputs, _tokenizerOptions);
    }

    private static BatchTokenizedResult BatchTokensToTensors(
        TokensView inputs,
        PretrainedTokenizerOptions tokenizerOptions)
    {
        int batchSize = inputs.Count;
        int maxTokenSize = inputs.MaxTokenSize;

        var result = CreateTokenAndMaskTensorsFromShape(batchSize, maxTokenSize);

        TensorDimensionSpan<long> tokenizationSpan = result.Tokens.GetDimensionSpan(0);
        TensorDimensionSpan<long> maskSpan = result.Mask.GetDimensionSpan(0);
        for (int i = 0; i < batchSize; i++)
        {
            TokenizeRow(tokenizerOptions, inputs[i], tokenizationSpan, maskSpan, i);
        }

        return result;
    }

    private static BatchTokenizedResult CreateTokenAndMaskTensorsFromShape(int batchSize, int maxTokenSize)
    {
        Span<nint> tensorShape = [batchSize, maxTokenSize];
        nint columnStride = maxTokenSize == 1 ? 0 : 1;
        Span<nint> strides = [maxTokenSize, columnStride];

        Tensor<long> tokenization = Tensor.Create<long>(tensorShape, strides); // would like to pool underlying array and use TensorMemory<T>
        Tensor<long> mask = Tensor.Create<long>(tensorShape, strides);
        return new BatchTokenizedResult(tokenization, mask);
    }

    private static void TokenizeRow(PretrainedTokenizerOptions tokenizerOptions, List<int> rowTokens, TensorDimensionSpan<long> tokenizationSpan,
        TensorDimensionSpan<long> maskSpan, int i)
    {
        Span<int> tokenizedInput = CollectionsMarshal.AsSpan(rowTokens);
        Span<long> tokenizationRowSpan = tokenizationSpan[i].AsSpan();
        Span<long> maskRowSpan = maskSpan[i].AsSpan();

        TensorPrimitives.ConvertChecked<int, long>(tokenizedInput, tokenizationRowSpan);

        if (tokenizerOptions.PaddingToken != 0) // No need - initialized to 0
        {
            tokenizationRowSpan[tokenizedInput.Length..].Fill(tokenizerOptions.PaddingToken);
        }

        maskRowSpan[..tokenizedInput.Length].Fill(1);
        // maskRow[tokenizedInput.Count..].Fill(0);  No need - initialized to 0
    }
}