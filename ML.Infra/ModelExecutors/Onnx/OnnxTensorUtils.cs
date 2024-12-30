using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace ML.Infra.ModelExecutors.Onnx;

public static class OnnxTensorUtils
{
    public static T[,] To2DArray<T>(this Tensor<T> tensor)
    {
        ReadOnlySpan<int> dims = tensor.Dimensions;
        var values = new T[dims[0], dims[1]];
        for (int i = 0; i < values.GetLength(0); i++)
        {
            for (int j = 0; j < values.GetLength(1); j++)
            {
                values[i, j] = tensor[i, j];
            }
        }

        return values;
    }

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