namespace ML.NLP.Tokenization;

public readonly ref struct TextView
{
    private readonly ReadOnlySpan<TokenizedText> _textInputs;

    public TextView(ReadOnlySpan<TokenizedText> textInputs)
    {
        _textInputs = textInputs;
    }

    public int Count => _textInputs.Length;

    public string this[int index] => _textInputs[index].Text;

    public void SetTokens(int index, List<int> tokens)
    {
        _textInputs[index].Tokens = tokens;
    }
}