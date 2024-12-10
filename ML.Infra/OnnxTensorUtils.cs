using CommunityToolkit.HighPerformance;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace ML.Infra;

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

    public static Tensor<T> ToTensor<T>(this Memory2D<T> memory2D)
    {
        ReadOnlySpan<int> dims = [memory2D.Height, memory2D.Width];
        if (memory2D.TryGetMemory(out var memory))
        {
            return new DenseTensor<T>(memory, dims);
        }

        var tensor = new DenseTensor<T>(dims);
        var span2D = memory2D.Span;
        for (int i = 0; i < memory2D.Height; i++)
        {
            for (int j = 0; j < memory2D.Width; j++)
            {
                tensor[i, j] = span2D[i, j];
            }
        }
        return tensor;
    }

    public static OrtValue[] ToOrtValues(this Span<Memory<long>> inputs, long[] dims)
    {
        var inputsOrts = new OrtValue[inputs.Length];
        for (int i = 0; i < inputs.Length; i++)
        {
            inputsOrts[i] = OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, inputs[i], dims);
        }
        return inputsOrts;
    }
}
