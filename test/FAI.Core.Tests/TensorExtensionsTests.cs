using System.Numerics.Tensors;
using FAI.Core;

namespace FAI.Core.Tests;

public class TensorExtensionsTests
{
    [Fact]
    public void AsSpan_TensorSpan_ReturnsCorrectSpan()
    {
        // Arrange
        float[] data = [1.0f, 2.0f, 3.0f];
        var tensorSpan = new TensorSpan<float>(data, [3]);

        // Act
        var span = tensorSpan.AsSpan();

        // Assert
        Assert.Equal(3, span.Length);
        Assert.Equal(1.0f, span[0]);
        Assert.Equal(2.0f, span[1]);
        Assert.Equal(3.0f, span[2]);

        // Verify it's a reference to the same data
        span[1] = 42.0f;
        Assert.Equal(42.0f, data[1]);
    }

    [Fact]
    public void AsSpan_ReadOnlyTensorSpan_ReturnsCorrectSpan()
    {
        // Arrange
        float[] data = [1.0f, 2.0f, 3.0f];
        var tensorSpan = new ReadOnlyTensorSpan<float>(data, [3]);

        // Act
        var span = tensorSpan.AsSpan();

        // Assert
        Assert.Equal(3, span.Length);
        Assert.Equal(1.0f, span[0]);
        Assert.Equal(2.0f, span[1]);
        Assert.Equal(3.0f, span[2]);
    }

    [Fact]
    public void AsMemory_Tensor_ReturnsCorrectMemory()
    {
        // Arrange
        float[] data = [1.0f, 2.0f, 3.0f];
        var tensor = Tensor.Create<float>(data, [3]);

        // Act
        var memory = tensor.AsMemory();

        // Assert
        Assert.Equal(3, memory.Length);
        Assert.Equal(1.0f, memory.Span[0]);
        Assert.Equal(2.0f, memory.Span[1]);
        Assert.Equal(3.0f, memory.Span[2]);

        // Verify reference
        memory.Span[1] = 42.0f;
        Assert.Equal(42.0f, data[1]);
    }
}
