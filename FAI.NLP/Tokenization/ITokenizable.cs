namespace FAI.NLP.Tokenization;

/// <summary>
/// Represents an entity that can be tokenized for natural language processing tasks. May represent multiple sentences.
/// </summary>
public interface ITokenizable
{
    /// <summary>
    /// Gets the total number of tokens present in the entity. Not including padding.
    /// </summary>
    int TokenCount { get; }

    /// <summary>
    /// Gets the maximum token length (of one of the sentences).
    /// </summary>
    int MaxTokenLength { get; }

    /// <summary>
    /// Gets the number of sentences or discrete text segments in the entity.
    /// </summary>
    int SentenceCount { get; }

    /// <summary>
    /// Tokenizes the entity using the provided pretrained tokenizer.
    /// </summary>
    /// <param name="pretrainedTokenizer">The tokenizer to use for tokenizing the entity.</param>
    void Tokenize(PretrainedTokenizer pretrainedTokenizer);
}