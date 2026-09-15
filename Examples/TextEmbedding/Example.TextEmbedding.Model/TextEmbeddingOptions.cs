using FAI.Core.Configurations;
using FAI.NLP.Configuration;
using FAI.Onnx.Configuration;

namespace Example.TextEmbedding.Model;

public sealed record TextEmbeddingModelOptions(
    string ModelDirectory,
    PretrainedTokenizerOptions TokenizerOptions,
    ModelExecutorType ModelExecutorType = ModelExecutorType.Simple,
    bool UseGpu = true)
{
    public TextEmbeddingOptions DecodingOptions { get; init; } = new(PoolingStrategy.Mean, Normalize: true, EmbeddingDimensions: 384);

    public TokenCountOrderingOptions TokenCountOrdering { get; init; } = new(Ascending: true);

    public MaxPaddedTokensPartitionerOptions MaxPaddedTokens { get; init; } = new(
        MaxPaddedTokenRatio: 0.1,
        MaxTokenCount: 2048);

    public ParallelPartitionSchedulerOptions ParallelScheduler { get; init; } = new(MaxConcurrency: 10);

    public static TextEmbeddingModelOptions Create(string modelDirectory)
        => new(modelDirectory, new PretrainedTokenizerOptions(MaxTokenLength: 256, IncludeTokenTypeIds: true));
}
