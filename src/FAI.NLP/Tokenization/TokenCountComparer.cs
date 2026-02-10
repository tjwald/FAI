namespace FAI.NLP.Tokenization;

/// <summary>
/// A comparer for tokenizable items, comparing them based on their maximum token length.
/// </summary>
/// <typeparam name="TToken">The type of tokenizable items to compare.</typeparam>
public readonly struct TokenCountComparer<TToken>(bool ascending) : IComparer<TToken> where TToken : ITokenizable
{
    private bool Ascending { get; } = ascending;

    /// <summary>
    /// Compares two tokenizable items based on their maximum token length.
    /// </summary>
    /// <returns>Compares based on MaxTokenLength.</returns>
    public int Compare(TToken? x, TToken? y)
    {
        int xCount = x!.MaxTokenLength;
        int yCount = y!.MaxTokenLength;

        return Ascending ? xCount.CompareTo(yCount) : yCount.CompareTo(xCount);
    }
}
