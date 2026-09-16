using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using FAI.Core;
using FAI.Core.Pipelines;
using FAI.NLP.Configuration;
using Microsoft.ML.Tokenizers;

namespace FAI.NLP.Tokenization;

/// <summary>
/// Represents a pretrained tokenizer used for tokenizing text inputs and managing token-related transformations.
/// Wraps a <see cref="Tokenizer"/> and adds batch functionality.
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
        if (truncationLength <= 0)
            return (tokenizedContextEnumerable, tokenizedTextEnumerable);

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

    public BatchEncode BatchTokenize(ReadOnlySpan<string> inputs)
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

    public BatchEncode BatchTokensToTensors(ReadOnlySpan<List<int>> inputs, int maxTokenSize)
    {
        return BatchTokensToTensors(inputs, _tokenizerOptions, maxTokenSize);
    }

    private static BatchEncode BatchTokensToTensors(ReadOnlySpan<List<int>> inputs, PretrainedTokenizerOptions tokenizerOptions, int maxTokenSize)
    {
        int batchSize = inputs.Length;
        var result = CreateTokenAndMaskTensorsFromShape(batchSize, maxTokenSize, tokenizerOptions.IncludeTokenTypeIds);

        TensorDimensionSpan<long> tokenizationSpan = result.InputIds.GetDimensionSpan(0);
        TensorDimensionSpan<long> maskSpan = result.AttentionMask.GetDimensionSpan(0);
        for (int i = 0; i < batchSize; i++)
        {
            TokenizeRow(tokenizerOptions, CollectionsMarshal.AsSpan(inputs[i]), tokenizationSpan, maskSpan, i);
        }

        return result;
    }

    public BatchEncode BatchTokensToTensors(TokensView inputs)
    {
        return BatchTokensToTensors(inputs, _tokenizerOptions);
    }

    private static BatchEncode BatchTokensToTensors(
        TokensView inputs,
        PretrainedTokenizerOptions tokenizerOptions)
    {
        int batchSize = inputs.Count;
        int maxTokenSize = inputs.MaxTokenSize;

        var result = CreateTokenAndMaskTensorsFromShape(batchSize, maxTokenSize, tokenizerOptions.IncludeTokenTypeIds);

        TensorDimensionSpan<long> tokenizationSpan = result.InputIds.GetDimensionSpan(0);
        TensorDimensionSpan<long> maskSpan = result.AttentionMask.GetDimensionSpan(0);
        for (int i = 0; i < batchSize; i++)
        {
            TokenizeRow(tokenizerOptions, inputs[i], tokenizationSpan, maskSpan, i);
        }

        return result;
    }

    private static BatchEncode CreateTokenAndMaskTensorsFromShape(int batchSize, int maxTokenSize, bool includeTokenTypeIds = false)
    {
        Span<nint> tensorShape = [batchSize, maxTokenSize];

        Tensor<long> tokenization = Tensor.CreateFromShape<long>(tensorShape); // would like to pool underlying array and use TensorMemory<T>
        Tensor<long> mask = Tensor.CreateFromShape<long>(tensorShape, tokenization.Strides);
        Tensor<long>? tokenTypeIds = includeTokenTypeIds
            ? Tensor.CreateFromShape<long>(tensorShape, tokenization.Strides)
            : null;
        return new BatchEncode(tokenization, mask, tokenTypeIds);
    }

    private static void TokenizeRow(PretrainedTokenizerOptions tokenizerOptions, ReadOnlySpan<int> rowTokens, TensorDimensionSpan<long> tokenizationSpan,
        TensorDimensionSpan<long> maskSpan, int i)
    {
        Span<long> tokenizationRowSpan = tokenizationSpan[i].AsSpan();
        Span<long> maskRowSpan = maskSpan[i].AsSpan();

        TensorPrimitives.ConvertChecked(rowTokens, tokenizationRowSpan);

        if (tokenizerOptions.PaddingToken != 0) // No need - initialized to 0
        {
            tokenizationRowSpan[rowTokens.Length..].Fill(tokenizerOptions.PaddingToken);
        }

        maskRowSpan[..rowTokens.Length].Fill(1);
        // maskRow[tokenizedInput.Count..].Fill(0);  No need - initialized to 0
    }
}
