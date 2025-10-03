namespace FAI.NLP.Tokenization;

/// <summary>
/// Represents a tokenized text input, storing raw text and its corresponding tokenized representation.
/// </summary>
/// <param name="Text">The original text to be tokenized.</param>
/// <param name="Tokens">
/// The tokenized representation of the text. Defaults to <c>null</c>, allowing deferred tokenization.
/// </param>
public sealed record TokenizedText(string Text, List<int>? Tokens = null) : ITokenizable
{
    /// <summary>
    /// Gets or sets the tokenized representation of the text.
    /// </summary>
    public List<int>? Tokens { get; set; } = Tokens;

    /// <summary>
    /// Gets the total number of tokens in the tokenized text.
    /// </summary>
    public int TokenCount => Tokens!.Count;

    /// <summary>
    /// Gets the maximum token length, which is equivalent to the token count.
    /// </summary>
    public int MaxTokenLength => TokenCount;

    /// <summary>
    /// Gets the number of sentences in the text. Returns 1 since this is a single sentence.
    /// </summary>
    public int SentenceCount => 1;

    /// <summary>
    /// Tokenizes the text using the provided pretrained tokenizer.
    /// Will do nothing if already tokenized.
    /// </summary>
    /// <param name="pretrainedTokenizer">The tokenizer to use for tokenizing the text.</param>
    public void Tokenize(PretrainedTokenizer pretrainedTokenizer)
    {
        Tokens ??= pretrainedTokenizer.Tokenize(Text);
    }

    /// <summary>
    /// Implicitly converts a raw text string into a <see cref="TokenizedText"/> instance.
    /// </summary>
    /// <param name="text">The text to convert into a tokenized format.</param>
    public static implicit operator TokenizedText(string text) => new(text);
}
