namespace FAI.NLP.Configuration;

/// <summary>
/// Specifies the pooling strategy used to convert token embeddings to sequence embeddings.
/// </summary>
public enum PoolingStrategy
{
    /// <summary>
    /// Computes the mean of non-padding token embeddings.
    /// </summary>
    Mean,

    /// <summary>
    /// Uses the first token embedding ([CLS]) as the sequence representation.
    /// </summary>
    ClsToken
}

/// <summary>
/// Represents configuration options for text embedding decoding.
/// </summary>
/// <param name="PoolingStrategy">The pooling strategy applied to token embeddings.</param>
/// <param name="Normalize">Whether to L2-normalize the resulting sequence embeddings.</param>
/// <param name="EmbeddingDimensions">Optional expected embedding dimensions. When null, discovered from model outputs.</param>
public sealed record TextEmbeddingOptions(
    PoolingStrategy PoolingStrategy = PoolingStrategy.Mean,
    bool Normalize = true,
    int? EmbeddingDimensions = null)
{
    public TextEmbeddingOptions() : this(PoolingStrategy.Mean, true, null)
    {
    }
}
