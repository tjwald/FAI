namespace FAI.NLP.Tokenization;

/// <summary>
/// Exposes token-count metadata for an already-tokenized NLP input. May represent multiple sentences.
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

}
