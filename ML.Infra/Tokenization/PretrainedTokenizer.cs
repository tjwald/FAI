using System.Buffers;
using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using CommunityToolkit.HighPerformance;
using Microsoft.ML.Tokenizers;
using MemoryExtensions = System.MemoryExtensions;

namespace ML.Infra.Tokenization;

public record PretrainedTokenizerOptions(int PaddingToken, int MaxTokenLength = 512);

public readonly record struct BatchTokenizedResult(Tensor<int> Tokens, Tensor<int> Mask)
{
    public int BatchSize => (int)Tokens.Lengths[0];
    public int MaxTokenCount => (int)Tokens.Lengths[1];
}

public class PretrainedTokenizer
{
    private readonly Tokenizer _tokenizer;
    private readonly PretrainedTokenizerOptions _tokenizerOptions;

    public PretrainedTokenizer(Tokenizer tokenizer, PretrainedTokenizerOptions tokenizerOptions)
    {
        _tokenizer = tokenizer;
        _tokenizerOptions = tokenizerOptions;
    }

    public BatchTokenizedResult BatchTokenize(ReadOnlySpan<string> inputs)
    {
        int maxTokenSize = 0;
        Span<IReadOnlyCollection<int>> tokenizedInputs = new IReadOnlyCollection<int>[inputs.Length];
        for (int i = 0; i < inputs.Length; i++)
        {
            IReadOnlyList<int> tokenizedInput = _tokenizer.EncodeToIds(inputs[i], _tokenizerOptions.MaxTokenLength, out _, out _);
            tokenizedInputs[i] = tokenizedInput;
            if (tokenizedInput.Count > maxTokenSize)
            {
                maxTokenSize = tokenizedInput.Count;
            }
        }

        Tensor<int> tokenization = Tensor.Create<int>([inputs.Length, maxTokenSize]);
        TensorSpan<int> tokenizationSpan = tokenization.AsTensorSpan();

        Tensor<int> mask = Tensor.Create<int>([inputs.Length, maxTokenSize]);
        TensorSpan<int> maskSpan = tokenization.AsTensorSpan();
        for (int i = 0; i < inputs.Length; i++)
        {
            Span<int> tokenizationRowSpan = tokenizationSpan.GetRowSpan(i);
            Span<int> maskRowSpan = maskSpan.GetRowSpan(i);
            IReadOnlyCollection<int> tokenizedInput = tokenizedInputs[i];
            
            foreach ((int j, int tokenId) in tokenizedInput.Index())
            {
                tokenizationRowSpan[j] = tokenId;
            }

            if (_tokenizerOptions.PaddingToken != 0) // No need - initialized to 0
            {
                tokenizationRowSpan[tokenizedInput.Count..].Fill(_tokenizerOptions.PaddingToken);
            }
            
            maskRowSpan[..tokenizedInput.Count].Fill(1);
            // maskRow[tokenizedInput.Count..].Fill(0);  No need - initialized to 0
        }

        return new BatchTokenizedResult(tokenization, mask);
    }
}