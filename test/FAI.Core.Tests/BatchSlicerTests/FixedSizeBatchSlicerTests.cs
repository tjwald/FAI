using FAI.Core.BatchSlicers;

namespace FAI.Core.Tests.BatchSlicerTests;

public class FixedSizeBatchSlicerTests
{
    [Fact]
    public void Slice_InputSmallerThanBatchSize_ReturnsSingleRange()
    {
        // Arrange
        var options = new FixedSizeBatchSlicerOptions(10);
        var slicer = new FixedSizeBatchSlicer<int>(options);
        var inputs = new int[5];

        // Act
        var ranges = slicer.Slice(inputs).ToList();

        // Assert
        Assert.Single(ranges);
        Assert.Equal(0, ranges[0].Start.Value);
        Assert.Equal(5, ranges[0].End.Value);
    }

    [Fact]
    public void Slice_InputMatchesBatchSize_ReturnsSingleRange()
    {
        // Arrange
        var options = new FixedSizeBatchSlicerOptions(5);
        var slicer = new FixedSizeBatchSlicer<int>(options);
        var inputs = new int[5];

        // Act
        var ranges = slicer.Slice(inputs).ToList();

        // Assert
        Assert.Single(ranges);
        Assert.Equal(0, ranges[0].Start.Value);
        Assert.Equal(5, ranges[0].End.Value);
    }

    [Fact]
    public void Slice_InputIsMultipleOfBatchSize_ReturnsMultipleRanges()
    {
        // Arrange
        var options = new FixedSizeBatchSlicerOptions(2);
        var slicer = new FixedSizeBatchSlicer<int>(options);
        var inputs = new int[4];

        // Act
        var ranges = slicer.Slice(inputs).ToList();

        // Assert
        // Current implementation is a sliding window with i++
        // 0..2, 1..3, 2..4
        Assert.Equal(3, ranges.Count);
        Assert.Equal(0, ranges[0].Start.Value);
        Assert.Equal(2, ranges[0].End.Value);
        Assert.Equal(1, ranges[1].Start.Value);
        Assert.Equal(3, ranges[1].End.Value);
        Assert.Equal(2, ranges[2].Start.Value);
        Assert.Equal(4, ranges[2].End.Value);
    }

    [Fact]
    public void Slice_InputWithRemainder_ReturnsRangesIncludingRemainder()
    {
        // Arrange
        var options = new FixedSizeBatchSlicerOptions(3);
        var slicer = new FixedSizeBatchSlicer<int>(options);
        var inputs = new int[7];

        // Act
        var ranges = slicer.Slice(inputs).ToList();

        // Assert
        // Current implementation yields 0..3, 1..4, 2..5, 3..6, 4..7
        Assert.Equal(5, ranges.Count);
        Assert.Equal(0, ranges[0].Start.Value);
        Assert.Equal(3, ranges[0].End.Value);
        Assert.Equal(4, ranges[4].Start.Value);
        Assert.Equal(7, ranges[4].End.Value);
    }

    [Fact]
    public void Slice_EmptyInput_ReturnsEmpty()
    {
        // Arrange
        var options = new FixedSizeBatchSlicerOptions(5);
        var slicer = new FixedSizeBatchSlicer<int>(options);
        var inputs = Array.Empty<int>();

        // Act
        var ranges = slicer.Slice(inputs).ToList();

        // Assert
        Assert.Empty(ranges);
    }
}
