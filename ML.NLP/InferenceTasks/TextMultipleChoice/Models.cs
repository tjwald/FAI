using ML.NLP.Tokenization;

namespace ML.NLP.InferenceTasks.TextMultipleChoice;

public sealed record TextMultipleChoiceInput(string Context, TokenizedText[] Choices) : ITokenizable
{
    private int _tokenCount = -1;
    private int _maxTokenLength = -1;

    public bool IsTokenized => Choices[0].Tokens is not null;

    public int TokenCount
    {
        get
        {
            if (_tokenCount == -1) throw new InvalidOperationException("Tokenized text is not tokenized.");
            
            return _tokenCount;
        }
    }

    public int MaxTokenLength
    {
        get
        {
            if (_maxTokenLength == -1) throw new InvalidOperationException("Tokenized text is not tokenized.");
            return _maxTokenLength;
        }
    }

    public int SentenceCount => Choices.Length;

    public void Tokenize(PretrainedTokenizer pretrainedTokenizer)
    {
        if (_tokenCount > 0) return;
        
        _tokenCount = 0;
        _maxTokenLength = 0;
        foreach (var text in Choices)
        {
            text.Tokenize(pretrainedTokenizer);
            _tokenCount += text.TokenCount;
            _maxTokenLength = Math.Max(_maxTokenLength, text.TokenCount);
        }
    }
}