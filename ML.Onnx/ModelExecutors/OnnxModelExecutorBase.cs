using System.Collections.Concurrent;
using System.Numerics.Tensors;
using Microsoft.ML.OnnxRuntime;
using ML.Infra;
using ML.Infra.Abstractions;
using ML.Infra.Configurations.ModelExecutors;
using ML.Onnx.Configuration;

namespace ML.Onnx.ModelExecutors;

public interface IOnnxModelExecutor<out T> : IModelExecutor<long, float> where T : IOnnxModelExecutor<T>
{
    static abstract T Create(InferenceSession session, RunOptions runOptions, OnnxModelExecutorOptions options);
}

public abstract class OnnxModelExecutorBase : IModelExecutor<long, float>
{
    protected readonly InferenceSession Session;
    protected readonly RunOptions RunOptions;
    private readonly ConcurrentBag<long[]> _dimensionsPool;
    private readonly ConcurrentBag<Memory<long>[]> _inputMemoryPool;

    protected OnnxModelExecutorBase(InferenceSession session, RunOptions runOptions)
    {
        Session = session;
        RunOptions = runOptions;
        _dimensionsPool = [];
        _inputMemoryPool = [];
    }

    public abstract Task<Tensor<float>[]> RunAsync(Tensor<long>[] inputs);

    protected OrtValue[] GetModelInputs(Tensor<long>[] inputs)
    {
        long[] dims = GetInputDims(inputs);

        Memory<long>[] modelInputs = GetInputsAsMemory(inputs);

        OrtValue[] ortValues = modelInputs.AsSpan().ToOrtValues(dims);

        //return to pool:
        _dimensionsPool.Add(dims);
        _inputMemoryPool.Add(modelInputs);

        return ortValues;
    }

    protected static Tensor<float>[] ToOutTensors(IReadOnlyCollection<OrtValue> result)
    {
        Tensor<float>[] outTensors = new Tensor<float>[result.Count];
        if (result is List<OrtValue> ortValues)
        {
            for (int i = 0; i < result.Count; i++)
            {
                OrtValue tensor = ortValues[i];
                outTensors[i] = ToOutTensor(tensor);
            }

            return outTensors;
        }

        int index = 0;
        foreach (OrtValue tensor in result)
        {
            outTensors[index] = ToOutTensor(tensor);
            index++;
        }

        return outTensors;
    }

    private static Tensor<float> ToOutTensor(OrtValue tensor)
    {
        long[] outDims = tensor.GetTensorTypeAndShape().Shape!;
        Span<nint> outDimsAsNInts = stackalloc nint[outDims.Length];
        Span<nint> strides = [outDims.Length, 1];
        for (int dim = 0; dim < outDims.Length; dim++)
        {
            outDimsAsNInts[dim] = (nint)outDims[dim];
        }

        Tensor<float> outTensor = Tensor.Create<float>(outDimsAsNInts, strides);
        tensor.GetTensorDataAsSpan<float>().CopyTo(outTensor.AsMemory().Span);
        tensor.Dispose();

        return outTensor;
    }

    private Memory<long>[] GetInputsAsMemory(Tensor<long>[] inputs)
    {
        Memory<long>[] modelInputs;
        if (!_inputMemoryPool.TryTake(out modelInputs!))
        {
            modelInputs = new Memory<long>[inputs.Length];
        }

        for (int i = 0; i < modelInputs.Length; i++)
        {
            modelInputs[i] = inputs[i].AsMemory();
        }

        return modelInputs;
    }

    private long[] GetInputDims(Tensor<long>[] inputs)
    {
        long[] dims;
        if (!_dimensionsPool.TryTake(out dims!))
        {
            dims = new long[inputs[0].Rank];
        }

        for (int i = 0; i < inputs[0].Rank; i++)
        {
            dims[i] = inputs[0].Lengths[i];
        }

        return dims;
    }
}