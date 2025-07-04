namespace FAI.NLP.Tokenization;

/// <summary>
/// A comparer for tokenizable items, comparing them based on their maximum token length.
/// </summary>
/// <typeparam name="TToken">The type of tokenizable items to compare.</typeparam>
public readonly struct TokenCountComparer<TToken> : IComparer<TToken> where TToken : ITokenizable
{
    /// <summary>
    /// Compares two tokenizable items based on their maximum token length.
    /// </summary>
    /// <returns>Compares based on MaxTokenLength.</returns>
    public int Compare(TToken? x, TToken? y)
    {
        int xCount = x!.MaxTokenLength;
        int yCount = y!.MaxTokenLength;

        return xCount.CompareTo(yCount);
    }
}