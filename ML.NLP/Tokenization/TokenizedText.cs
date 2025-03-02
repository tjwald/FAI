namespace ML.NLP.Tokenization;

public sealed record TokenizedText(string Text, List<int>? Tokens = null)
{
    public List<int>? Tokens { get; set; } = Tokens;
    public int Count => Tokens!.Count;
    
    public static implicit operator TokenizedText(string text) => new(text);
}