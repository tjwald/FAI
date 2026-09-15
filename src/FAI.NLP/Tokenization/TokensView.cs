namespace FAI.NLP.Tokenization;

/// <summary>
/// Represents a readonly view of tokenized text inputs, providing access to token data and size calculations.
/// </summary>
public ref struct TokensView
{
    private readonly ReadOnlySpan<TokenizedText> _textInputs;
    private int _maxTokens;

    /// <summary>
    /// Initializes a new instance of the <see cref="TokensView"/> struct with the given tokenized text inputs.
    /// </summary>
    /// <param name="textInputs">The span of tokenized text inputs to wrap.</param>
    public TokensView(ReadOnlySpan<TokenizedText> textInputs)
    {
        _textInputs = textInputs;
        _maxTokens = -1;
    }

    /// <summary>
    /// Gets the number of tokenized text inputs in the view.
    /// </summary>
    public int Count => _textInputs.Length;

    /// <summary>
    /// Gets the maximum token size among all tokenized text inputs in the view.
    /// </summary>
    public int MaxTokenSize
    {
        get
        {
            if (_maxTokens >= 0)
                return _maxTokens;

            int maxTokens = -1;
            foreach (var t in _textInputs)
            {
                maxTokens = Math.Max(maxTokens, t.Tokens.Length);
            }

            _maxTokens = maxTokens;

            return _maxTokens;
        }
    }

    /// <summary>
    /// Gets the tokenized representation of the text input at the specified index.
    /// </summary>
    /// <param name="index">The index of the tokenized text input to retrieve.</param>
    /// <returns>The tokenized representation of the text input.</returns>
    public ReadOnlySpan<int> this[int index] => _textInputs[index].Tokens.Span;
}
