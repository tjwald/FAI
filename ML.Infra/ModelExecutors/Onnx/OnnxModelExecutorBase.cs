using System.Numerics.Tensors;
using Microsoft.ML.OnnxRuntime;
using ML.Infra.Abstractions;

namespace ML.Infra.ModelExecutors.Onnx;

public interface IOnnxModelExecutor<out T> : IModelExecutor<long, float> where T: IOnnxModelExecutor<T>
{
    static abstract T Create(InferenceSession session, RunOptions runOptions, OnnxModelExecutorOptions options);
}

public abstract class OnnxModelExecutorBase : IModelExecutor<long, float>
{
    protected readonly InferenceSession Session;
    protected readonly RunOptions RunOptions;

    protected OnnxModelExecutorBase(InferenceSession session, RunOptions runOptions)
    {
        Session = session;
        RunOptions = runOptions;
    }

    public abstract Task<Tensor<float>[]> RunAsync(Tensor<long>[] inputs);

    protected static OrtValue[] GetModelInputs(Tensor<long>[] inputs)
    {
        long[] dims = GetInputDims(inputs);

        Span<Memory<long>> modelInputs = GetInputsAsMemory(inputs);

        OrtValue[] ortValues = modelInputs.ToOrtValues(dims);
        return ortValues;
    }

    protected static Tensor<float>[] ToOutTensors(IReadOnlyCollection<OrtValue> result)
    {
        var outTensors = new Tensor<float>[result.Count];
        foreach ((int i, OrtValue tensor) in result.Index())
        {
            long[] outDims = tensor.GetTensorTypeAndShape().Shape!;
            nint[] outDimsAsNInts = new nint[outDims.Length];
            for (int dim = 0; dim < outDims.Length; dim++)
            {
                outDimsAsNInts[dim] = (nint)outDims[dim];
            }

            Tensor<float> outTensor = Tensor.Create<float>(outDimsAsNInts);
            tensor.GetTensorDataAsSpan<float>().CopyTo(outTensor.AsMemory().Span);
            outTensors[i] = outTensor;
            tensor.Dispose();
        }

        return outTensors;
    }

    private static Span<Memory<long>> GetInputsAsMemory(Tensor<long>[] inputs)
    {
        Span<Memory<long>> modelInputs = new Memory<long>[inputs.Length];
        for (int i = 0; i < modelInputs.Length; i++)
        {
            modelInputs[i] = inputs[i].AsMemory();
        }

        return modelInputs;
    }

    private static long[] GetInputDims(Tensor<long>[] inputs)
    {
        long[] dims = new long[inputs[0].Rank];
        for (int i = 0; i < inputs[0].Rank; i++)
        {
            dims[i] = inputs[0].Lengths[i];
        }

        return dims;
    }
}