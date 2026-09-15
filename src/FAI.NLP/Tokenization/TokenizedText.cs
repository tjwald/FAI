namespace FAI.NLP.Tokenization;

/// <summary>
/// Represents immutable tokenized text.
/// </summary>
/// <param name="Text">The original text to be tokenized.</param>
/// <param name="Tokens">
/// The tokenized representation of the text.
/// </param>
public sealed record TokenizedText(string Text, ReadOnlyMemory<int> Tokens) : ITokenizable
{
    /// <summary>
    /// Gets the total number of tokens in the tokenized text.
    /// </summary>
    public int TokenCount => Tokens.Length;

    /// <summary>
    /// Gets the maximum token length, which is equivalent to the token count.
    /// </summary>
    public int MaxTokenLength => TokenCount;

    /// <summary>
    /// Gets the number of sentences in the text. Returns 1 since this is a single sentence.
    /// </summary>
    public int SentenceCount => 1;

}
