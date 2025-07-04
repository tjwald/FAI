using Microsoft.ML.OnnxRuntime;

namespace FAI.Onnx.Utils;

/// <summary>
/// Provides utility methods for working with ONNX tensors.
/// </summary>
internal static class OnnxTensorUtils
{
    /// <summary>
    /// Converts a span of memory buffers into an array of ONNX runtime tensor values (<see cref="OrtValue"/>).
    /// </summary>
    /// <typeparam name="T">The data type of the elements in the tensor. Must be an unmanaged type.</typeparam>
    /// <param name="inputs">A span of memory buffers containing the input data.</param>
    /// <param name="dims">The dimensions of the tensors.</param>
    /// <returns>An array of <see cref="OrtValue"/> representing the converted tensor values.</returns>
    public static OrtValue[] ToOrtValues<T>(this Span<Memory<T>> inputs, long[] dims) where T : unmanaged
    {
        var inputsOrts = new OrtValue[inputs.Length];
        for (int i = 0; i < inputs.Length; i++)
        {
            inputsOrts[i] = OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, inputs[i], dims);
        }

        return inputsOrts;
    }
}