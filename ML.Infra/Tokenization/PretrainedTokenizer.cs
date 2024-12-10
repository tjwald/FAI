using CommunityToolkit.HighPerformance;
using Microsoft.ML.Tokenizers;

namespace ML.Infra.Tokenization;

public record PretrainedTokenizerOptions(int PaddingToken, int MaxTokenLength = 512);

public readonly record struct BatchTokenizedResult(int[,] Tokens, int[,] Mask)
{
    public int BatchSize => Tokens.GetLength(0);
    public int MaxTokenCount => Tokens.GetLength(1);
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

        int[,] tokenization = new int[inputs.Length, maxTokenSize];
        Span2D<int> tokenization2DSpan = tokenization.AsSpan2D();

        int[,] mask = new int[inputs.Length, maxTokenSize];
        Span2D<int> mask2DSpan = tokenization.AsSpan2D();
        for (int i = 0; i < inputs.Length; i++)
        {
            IReadOnlyCollection<int> tokenizedInput = tokenizedInputs[i];
            Span<int> tokenizationRow = tokenization2DSpan.GetRowSpan(i);
            
            foreach ((int j, int tokenId) in tokenizedInput.Index())
            {
                tokenizationRow[j] = tokenId;
            }

            if (_tokenizerOptions.PaddingToken != 0) // No need - initialized to 0
            {
                tokenizationRow[tokenizedInput.Count..].Fill(_tokenizerOptions.PaddingToken);
            }
            
            Span<int> maskRow = mask2DSpan.GetRowSpan(i);
            maskRow[..tokenizedInput.Count].Fill(1);
            // maskRow[tokenizedInput.Count..].Fill(0);  No need - initialized to 0
        }

        return new BatchTokenizedResult(tokenization, mask);
    }
}