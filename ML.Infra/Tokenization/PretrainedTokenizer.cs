using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using Microsoft.ML.Tokenizers;

namespace ML.Infra.Tokenization;

public sealed record PretrainedTokenizerOptions(int PaddingToken, int MaxTokenLength = 512);

public readonly record struct BatchTokenizedResult(Tensor<long> Tokens, Tensor<long> Mask)
{
    public int BatchSize => (int)Tokens.Lengths[0];
    public int MaxTokenCount => (int)Tokens.Lengths[1];
}

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

        (Tensor<long> tokenization, Tensor<long> mask) = BatchTokensToTensors(tokenizedInputs, _tokenizerOptions, maxTokenSize);

        return new BatchTokenizedResult(tokenization, mask);
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

        (Tensor<long> tokenization, Tensor<long> mask) = BatchTokensToTensors(tokenizedInputs, _tokenizerOptions, maxTokenSize);

        return new BatchTokenizedResult(tokenization, mask);
    }

    public (Tensor<long> tokens, Tensor<long> mask) BatchTokensToTensors(
        ReadOnlySpan<List<int>> inputs,
        int maxTokenSize)
    {
        return BatchTokensToTensors(inputs, _tokenizerOptions, maxTokenSize);
    }

    private static (Tensor<long> tokens, Tensor<long> mask) BatchTokensToTensors(
        ReadOnlySpan<List<int>> inputs,
        PretrainedTokenizerOptions tokenizerOptions,
        int maxTokenSize)
    {
        Span<nint> tensorShape = [inputs.Length, maxTokenSize];
        Span<nint> strides = [maxTokenSize, 1];
        Tensor<long> tokenization = Tensor.Create<long>(tensorShape, strides); // would like to pool underlying array and use TensorMemory<T>
        TensorSpan<long> tokenizationSpan = tokenization.AsTensorSpan();

        Tensor<long> mask = Tensor.Create<long>(tensorShape, strides);
        TensorSpan<long> maskSpan = mask.AsTensorSpan();
        for (int i = 0; i < inputs.Length; i++)
        {
            Span<long> tokenizationRowSpan = tokenizationSpan.GetRowSpan(i);
            Span<long> maskRowSpan = maskSpan.GetRowSpan(i);
            Span<int> tokenizedInput = CollectionsMarshal.AsSpan(inputs[i]);

            for (int j = 0; j < tokenizedInput.Length; j++)
            {
                tokenizationRowSpan[j] = tokenizedInput[j];
            }

            if (tokenizerOptions.PaddingToken != 0) // No need - initialized to 0
            {
                tokenizationRowSpan[tokenizedInput.Length..].Fill(tokenizerOptions.PaddingToken);
            }

            maskRowSpan[..tokenizedInput.Length].Fill(1);
            // maskRow[tokenizedInput.Count..].Fill(0);  No need - initialized to 0
        }

        return (tokenization, mask);
    }
    
    public (Tensor<long> tokens, Tensor<long> mask) BatchTokensToTensors(TokensView inputs)
    {
        return BatchTokensToTensors(inputs, _tokenizerOptions);
    }

    private static (Tensor<long> tokens, Tensor<long> mask) BatchTokensToTensors(
        TokensView inputs,
        PretrainedTokenizerOptions tokenizerOptions)
    {
        Span<nint> tensorShape = [inputs.Count, inputs.MaxTokenSize];
        Span<nint> strides = [inputs.MaxTokenSize, 1];
        Tensor<long> tokenization = Tensor.Create<long>(tensorShape, strides); // would like to pool underlying array and use TensorMemory<T>
        TensorSpan<long> tokenizationSpan = tokenization.AsTensorSpan();

        Tensor<long> mask = Tensor.Create<long>(tensorShape, strides);
        TensorSpan<long> maskSpan = mask.AsTensorSpan();
        for (int i = 0; i < inputs.Count; i++)
        {
            Span<long> tokenizationRowSpan = tokenizationSpan.GetRowSpan(i);
            Span<long> maskRowSpan = maskSpan.GetRowSpan(i);
            Span<int> tokenizedInput = CollectionsMarshal.AsSpan(inputs[i]);

            for (int j = 0; j < tokenizedInput.Length; j++)
            {
                tokenizationRowSpan[j] = tokenizedInput[j];
            }

            if (tokenizerOptions.PaddingToken != 0) // No need - initialized to 0
            {
                tokenizationRowSpan[tokenizedInput.Length..].Fill(tokenizerOptions.PaddingToken);
            }

            maskRowSpan[..tokenizedInput.Length].Fill(1);
            // maskRow[tokenizedInput.Count..].Fill(0);  No need - initialized to 0
        }

        return (tokenization, mask);
    }
}