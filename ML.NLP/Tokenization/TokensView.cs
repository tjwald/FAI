namespace ML.NLP.Tokenization;

public ref struct TokensView
{
    private readonly ReadOnlySpan<TokenizedText> _textInputs;
    private int _maxTokens;

    public TokensView(ReadOnlySpan<TokenizedText> textInputs)
    {
        _textInputs = textInputs;
        _maxTokens = -1;
    }

    public int Count => _textInputs.Length;

    public int MaxTokenSize
    {
        get
        {
            if (_maxTokens >= 0) return _maxTokens;
            
            int maxTokens = -1;
            foreach (var t in _textInputs)
            {
                maxTokens = Math.Max(maxTokens, t.Tokens!.Count);
            }
            _maxTokens = maxTokens;

            return _maxTokens;
        }
    }

    public List<int> this[int index] => _textInputs[index].Tokens!;
}