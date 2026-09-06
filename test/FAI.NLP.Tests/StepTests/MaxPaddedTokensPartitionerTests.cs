using FAI.NLP.Configuration;
using FAI.NLP.Pipelines;
using FAI.NLP.Tokenization;

namespace FAI.NLP.Tests.PipelineTests;

public class MaxPaddedTokensPartitionerTests
{
    public record TestTokenizable(int TokenCount) : ITokenizable
    {
        public int MaxTokenLength => TokenCount;
        public int SentenceCount => 1;
    }

    [Fact]
    public void Partition_UsesMaxPaddedTokenRules()
    {
        var options = new MaxPaddedTokensPartitionerOptions(MaxPaddedTokenRatio: 0.5, MaxTokenCount: 10);
        var partitioner = new MaxPaddedTokensPartitioner<TestTokenizable>(options);
        ReadOnlyMemory<TestTokenizable> inputs = new TestTokenizable[] { new(4), new(4), new(4) };

        Range[] ranges = partitioner.Partition(inputs).ToArray();

        Assert.Equal([0..2, 2..3], ranges);
    }

    [Fact]
    public void Partition_DescendingOrder_RetainsMaximumTokenLength()
    {
        var options = new MaxPaddedTokensPartitionerOptions(MaxPaddedTokenRatio: 1.0, MaxTokenCount: 10);
        var partitioner = new MaxPaddedTokensPartitioner<TestTokenizable>(options);
        ReadOnlyMemory<TestTokenizable> inputs = new TestTokenizable[] { new(6), new(2) };

        Range[] ranges = partitioner.Partition(inputs).ToArray();

        // 2 items with max length 6 would require 2 * 6 = 12 padded tokens, which exceeds 10.
        Assert.Equal([0..1, 1..2], ranges);
    }

    public record MultipleChoiceTokenizable(int TokenCount, int MaxTokenLength, int SentenceCount) : ITokenizable;

    [Fact]
    public void Partition_MultipleSentences_CountsAllSentencesFromInitialItem()
    {
        var options = new MaxPaddedTokensPartitionerOptions(MaxPaddedTokenRatio: 1.0, MaxTokenCount: 30);
        var partitioner = new MaxPaddedTokensPartitioner<MultipleChoiceTokenizable>(options);
        // Each item has 4 sentences with max token length 5 (each item = 20 padded tokens)
        ReadOnlyMemory<MultipleChoiceTokenizable> inputs = new MultipleChoiceTokenizable[]
        {
            new(TokenCount: 20, MaxTokenLength: 5, SentenceCount: 4),
            new(TokenCount: 20, MaxTokenLength: 5, SentenceCount: 4),
        };

        Range[] ranges = partitioner.Partition(inputs).ToArray();

        // 2 items * 4 sentences = 8 sentences * 5 = 40 padded tokens, which exceeds 30.
        Assert.Equal([0..1, 1..2], ranges);
    }
}
