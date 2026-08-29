using FAI.Core.Configurations;
using FAI.NLP.Configuration;
using FAI.Onnx.Configuration;

namespace Example.TextEmbedding.Model;

public sealed record TextEmbeddingOptions(
    string ModelDirectory,
    PretrainedTokenizerOptions TokenizerOptions,
    ModelExecutorType ModelExecutorType = ModelExecutorType.Simple,
    bool UseGpu = true)
{
    public TokenCountOrderingOptions TokenCountOrdering { get; init; } = new(Ascending: true);

    public MaxPaddedTokensPartitionerOptions MaxPaddedTokens { get; init; } = new(
        MaxPaddedTokenRatio: 0.1,
        MaxTokenCount: 2048);

    public ParallelPartitionSchedulerOptions ParallelScheduler { get; init; } = new(MaxConcurrency: 8);

    public static TextEmbeddingOptions Create(string modelDirectory)
        => new(modelDirectory, new PretrainedTokenizerOptions(MaxTokenLength: 256));
}
