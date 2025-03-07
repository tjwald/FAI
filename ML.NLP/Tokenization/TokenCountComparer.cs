namespace ML.NLP.Tokenization;

public readonly struct TokenCountComparer<TToken> : IComparer<TToken> where TToken: ITokenizable
{
    public int Compare(TToken? x, TToken? y)
    {
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(y);

        int xCount = x.MaxTokenLength;
        int yCount = y.MaxTokenLength;

        return xCount.CompareTo(yCount);
    }
}