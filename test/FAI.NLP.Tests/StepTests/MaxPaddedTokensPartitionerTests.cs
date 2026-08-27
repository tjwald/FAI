using FAI.NLP.Configuration;
using FAI.NLP.Steps;
using FAI.NLP.Tokenization;

namespace FAI.NLP.Tests.StepTests;

public class MaxPaddedTokensPartitionerTests
{
    public record TestTokenizable(int TokenCount) : ITokenizable
    {
        public int MaxTokenLength => TokenCount;
        public int SentenceCount => 1;
        public void Tokenize(PretrainedTokenizer tokenizer) { }
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
}
