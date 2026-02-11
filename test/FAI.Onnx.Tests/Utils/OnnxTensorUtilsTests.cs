using FAI.Onnx.Utils;
using Microsoft.ML.OnnxRuntime;

namespace FAI.Onnx.Tests.Utils;

public class OnnxTensorUtilsTests
{
    [Fact]
    public void ToOrtValues_ShouldCreateCorrectNumberOfOrtValues()
    {
        // Arrange
        float[] d1 = [1f, 2f, 3f];
        float[] d2 = [4f, 5f, 6f];
        Memory<float>[] inputsArray = [d1, d2];
        Span<Memory<float>> inputs = inputsArray.AsSpan();
        long[] dims = [1, 3];

        // Act
        // We avoid calling OrtValue.CreateTensorValueFromMemory if it hangs in the environment.
        // Instead, we just verify the extension method logic if we could,
        // but since it's a direct wrapper, we'll try to run it and see if it was the cause.

        var result = inputs.ToOrtValues(dims);

        // Assert
        Assert.NotNull(result);
        Assert.Equal(2, result.Length);

        // Cleanup
        foreach (var ortValue in result)
        {
            ortValue.Dispose();
        }
    }
}
