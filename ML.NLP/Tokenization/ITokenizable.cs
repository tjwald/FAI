namespace ML.NLP.Tokenization;

public interface ITokenizable
{
    int TokenCount { get; }
    int MaxTokenLength { get; }
    int SentenceCount { get; }
    void Tokenize(PretrainedTokenizer pretrainedTokenizer);
}