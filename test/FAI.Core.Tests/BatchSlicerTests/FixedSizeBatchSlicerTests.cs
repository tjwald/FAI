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
        Assert.Equal([0..5], ranges);
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
        Assert.Equal([0..5], ranges);
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
        Assert.Equal([0..2, 1..3, 2..4], ranges);
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
        Assert.Equal([0..3, 1..4, 2..5, 3..6, 4..7], ranges);
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
