using System.Collections.Concurrent;
using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;
using ML.Infra;
using ML.Infra.Abstractions;
using ML.Infra.Utilities;
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
    private readonly SemaphoreSlim? _semaphore;


    protected OnnxModelExecutorBase(InferenceSession session, RunOptions runOptions, int? maxThreads = null)
    {
        Session = session;
        RunOptions = runOptions;
        _dimensionsPool = [];
        _inputMemoryPool = [];
        _semaphore = maxThreads.HasValue ? new SemaphoreSlim(maxThreads.Value, maxThreads.Value) : null;
        
    }

    protected abstract Task<IDisposableReadOnlyCollection<OrtValue>> RunSessionInference(Tensor<long>[] inputs, OrtValue[] ortValues);

    private async Task<IDisposableReadOnlyCollection<OrtValue>> ExecuteModelAsync(Tensor<long>[] inputs)
    {
        OrtValue[] ortValues = GetModelInputs(inputs);

        IDisposableReadOnlyCollection<OrtValue> result;
        using (await _semaphore.EnterScope())
        {
            result = await RunSessionInference(inputs, ortValues);
        }

        foreach (var input in ortValues)
        {
            input.Dispose();
        }
        return result;
    }

    public async Task<Tensor<float>[]> RunAsync(Tensor<long>[] inputs)
    {
        using IDisposableReadOnlyCollection<OrtValue> result = await ExecuteModelAsync(inputs);

        return ToOutTensors(result);
    }

    public async Task RunAsync(Tensor<long>[] inputs, Action<ReadOnlyTensorSpan<float>, int> postProcess)
    {
        using IDisposableReadOnlyCollection<OrtValue> result = await ExecuteModelAsync(inputs);
        for (int i = 0; i < result.Count; i++)
        {
            postProcess(result[i].GetTensorDataAsTensorSpan<float>(), i);
        }
    }

    protected virtual OrtValue[] GetModelInputs(Tensor<long>[] inputs)
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
        var outTensors = new Tensor<float>[result.Count];
        if (result is List<OrtValue> ortValues)
        {
            Span<OrtValue> span = CollectionsMarshal.AsSpan(ortValues);
            for (int i = 0; i < span.Length; i++)
            {
                outTensors[i] = ToOutTensor(span[i]);
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
        ReadOnlyTensorSpan<float> x = tensor.GetTensorDataAsTensorSpan<float>();
        
        Tensor<float> outTensor = Tensor.Create<float>(x.Lengths, x.Strides);
        x.CopyTo(outTensor);
        
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