using FAI.NLP.BatchSlicer;
using FAI.NLP.Configuration.PipelineBatchExecutors;
using FAI.NLP.Tokenization;

namespace FAI.NLP.Tests.BatchSlicer;

public class MaxPaddedTokensBatchSlicerTests
{
    public record TestTokenizable(int TokenCount) : ITokenizable
    {
        public int MaxTokenLength => TokenCount;
        public int SentenceCount => 1;
        public void Tokenize(PretrainedTokenizer tokenizer) { }
    }

    [Fact]
    public void Slice_RespectsMaxTokenCount()
    {
        // Arrange
        var options = new MaxPaddedTokensSlicerOptions
        {
            MaxTokenCount = 10,
            MaxPaddedTokenRatio = 0.5 // Allow up to 50% padding
        };
        var slicer = new MaxPaddedTokensBatchSlicer<TestTokenizable>(options);

        // Items with token counts: 4, 4, 4
        // Batch 1: (4+4) * 2 = 8 <= 10. OK.
        // Batch 1 + next: (4+4+4) * 3 = 12 > 10. Break.
        TestTokenizable[] inputs = [new(4), new(4), new(4)];

        // Act
        var ranges = slicer.Slice(inputs).ToList();

        // Assert
        Assert.Equal(2, ranges.Count);
        Assert.Equal(0..2, ranges[0]);
        Assert.Equal(2..3, ranges[1]);
    }

    [Fact]
    public void Slice_RespectsMaxPaddedTokenRatio()
    {
        // Arrange
        var options = new MaxPaddedTokensSlicerOptions
        {
            MaxTokenCount = 100,
            MaxPaddedTokenRatio = 0.1 // Only 10% padding allowed
        };
        var slicer = new MaxPaddedTokensBatchSlicer<TestTokenizable>(options);

        // Slicer assumes input is sorted ASCENDING by MaxTokenLength.
        // Item 1: 2 tokens.
        // Item 2: 10 tokens.
        // If batched: MaxLen = 10, Count = 2. Padded = 20. Sum = 12.
        // factor = 1.0 - 0.1 = 0.9.
        // newSum < newPadded * factor <=> 12 < 20 * 0.9 (18) => TRUE. BREAKS.
        TestTokenizable[] inputs = [new(2), new(10)];

        // Act
        var ranges = slicer.Slice(inputs).ToList();

        // Assert
        Assert.Equal(2, ranges.Count);
        Assert.Equal(0..1, ranges[0]);
        Assert.Equal(1..2, ranges[1]);
    }
}
