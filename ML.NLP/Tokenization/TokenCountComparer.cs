namespace ML.NLP.Tokenization;

public readonly struct TokenCountComparer : IComparer<TokenizedText>
{
    private readonly PretrainedTokenizer _tokenizer;

    public TokenCountComparer(PretrainedTokenizer tokenizer)
    {
        _tokenizer = tokenizer;
    }

    public int Compare(TokenizedText? x, TokenizedText? y)
    {
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(y);

        x.Tokens ??= _tokenizer.Tokenize(x.Text);
        y.Tokens ??= _tokenizer.Tokenize(y.Text);

        int xCount = x.Count;
        int yCount = y.Count;

        return xCount.CompareTo(yCount);
    }
}