using FAI.NLP.Tokenization;

namespace FAI.NLP.InferenceTasks.TextMultipleChoice;

/// <summary>
/// Represents the input for a multiple-choice text classification task.
/// </summary>
/// <param name="Context">The context or prompt that applies to the multiple-choice options.</param>
/// <param name="Choices">The possible choices for classification, represented as tokenized text.</param>
public record TextMultipleChoiceInput(string Context, TokenizedText[] Choices) : ITokenizable
{
    private int _tokenCount = -1;
    private int _maxTokenLength = -1;

    /// <summary>
    /// Indicates whether the input choices have been tokenized.
    /// </summary>
    public bool IsTokenized => Choices[0].Tokens is not null;

    /// <summary>
    /// Gets the total number of tokens across all choices.
    /// Throws an exception if the input has not been tokenized.
    /// </summary>
    /// <exception cref="InvalidOperationException">Thrown if the input text has not been tokenized.</exception>
    public int TokenCount
    {
        get
        {
            if (_tokenCount == -1) throw new InvalidOperationException("Tokenized text is not tokenized.");
            return _tokenCount;
        }
    }

    /// <summary>
    /// Gets the maximum token length among all choices.
    /// Throws an exception if the input has not been tokenized.
    /// </summary>
    /// <exception cref="InvalidOperationException">Thrown if the input text has not been tokenized.</exception>
    public int MaxTokenLength
    {
        get
        {
            if (_maxTokenLength == -1) throw new InvalidOperationException("Tokenized text is not tokenized.");
            return _maxTokenLength;
        }
    }

    /// <summary>
    /// Gets the number of choices available for classification.
    /// </summary>
    public int SentenceCount => Choices.Length;

    /// <summary>
    /// Tokenizes the multiple-choice options using the specified pretrained tokenizer.
    /// If the input is already tokenized, this method does nothing.
    /// </summary>
    /// <param name="pretrainedTokenizer">The tokenizer to use for tokenizing the choices.</param>
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