namespace ML.NLP.Tokenization;

public sealed record TokenizedText(string Text, List<int>? Tokens = null) : ITokenizable
{
    public List<int>? Tokens { get; set; } = Tokens;
    public int TokenCount => Tokens!.Count;
    
    public int MaxTokenLength => TokenCount;
    public int SentenceCount => 1;

    public void Tokenize(PretrainedTokenizer pretrainedTokenizer)
    {
        Tokens ??= pretrainedTokenizer.Tokenize(Text);
    }

    public static implicit operator TokenizedText(string text) => new(text);
}