using System.Numerics.Tensors;
using Microsoft.ML.Tokenizers;

namespace ML.Infra.Tokenization;

public record PretrainedTokenizerOptions(int PaddingToken, int MaxTokenLength = 512);

public readonly record struct BatchTokenizedResult(Tensor<long> Tokens, Tensor<long> Mask)
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

    public Tokenizer Tokenizer => _tokenizer;
    
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

        Tensor<long> tokenization = Tensor.Create<long>([inputs.Length, maxTokenSize]);  // would like to pool underlying array and use TensorMemory<T>
        TensorSpan<long> tokenizationSpan = tokenization.AsTensorSpan();

        Tensor<long> mask = Tensor.Create<long>([inputs.Length, maxTokenSize]);
        TensorSpan<long> maskSpan = mask.AsTensorSpan();
        for (int i = 0; i < inputs.Length; i++)
        {
            Span<long> tokenizationRowSpan = tokenizationSpan.GetRowSpan(i);
            Span<long> maskRowSpan = maskSpan.GetRowSpan(i);
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