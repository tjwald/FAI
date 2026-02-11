using FAI.Onnx.Utils;
using Microsoft.ML.OnnxRuntime;

namespace FAI.Onnx.Tests.Utils;

public class OnnxTensorUtilsTests
{
    [Fact]
    public void ToOrtValues_ShouldCreateCorrectNumberOfOrtValues()
    {
        // Arrange
        var data1 = new float[] { 1, 2, 3 }.AsMemory();
        var data2 = new float[] { 4, 5, 6 }.AsMemory();
        Memory<float>[] inputsArray = [data1, data2];
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
