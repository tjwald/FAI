namespace FAI.NLP.Tokenization;

/// <summary>
/// Provides a readonly view over a span of tokenized text inputs, allowing access to text content and token assignment.
/// </summary>
public readonly ref struct TextView
{
    private readonly ReadOnlySpan<TokenizedText> _textInputs;

    /// <summary>
    /// Initializes a new instance of the <see cref="TextView"/> struct with the given tokenized text inputs.
    /// </summary>
    /// <param name="textInputs">The span of tokenized text inputs to wrap.</param>
    public TextView(ReadOnlySpan<TokenizedText> textInputs)
    {
        _textInputs = textInputs;
    }

    /// <summary>
    /// Gets the number of tokenized text inputs in the view.
    /// </summary>
    public int Count => _textInputs.Length;

    /// <summary>
    /// Gets the raw text content of the tokenized input at the specified index.
    /// </summary>
    /// <param name="index">The index of the tokenized text input to retrieve.</param>
    /// <returns>The raw text string of the tokenized input.</returns>
    public string this[int index] => _textInputs[index].Text;

    /// <summary>
    /// Assigns tokenized values to a specific text input at the given index.
    /// </summary>
    /// <param name="index">The index of the tokenized text input to modify.</param>
    /// <param name="tokens">The tokenized representation to assign.</param>
    public void SetTokens(int index, List<int> tokens)
    {
        _textInputs[index].Tokens = tokens;
    }
}
