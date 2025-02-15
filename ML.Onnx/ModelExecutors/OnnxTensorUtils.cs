using Microsoft.ML.OnnxRuntime;

namespace ML.Onnx.ModelExecutors;

public static class OnnxTensorUtils
{
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