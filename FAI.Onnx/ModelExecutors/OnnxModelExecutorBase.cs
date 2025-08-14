using System.Collections.Concurrent;
using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using FAI.Core;
using FAI.Core.Abstractions;
using FAI.Core.Utilities;
using FAI.Onnx.Configuration;
using FAI.Onnx.Utils;
using Microsoft.ML.OnnxRuntime;

namespace FAI.Onnx.ModelExecutors;

/// <summary>
/// Defines a contract for an ONNX model executor.
/// </summary>
/// <typeparam name="T">The type of the executor implementing this interface.</typeparam>
public interface IOnnxModelExecutor<out T> : IModelExecutor<long, float> where T : IOnnxModelExecutor<T>
{
    /// <summary>
    /// Creates an instance of the ONNX model executor with the specified session, options, and configuration.
    /// </summary>
    /// <param name="session">The ONNX runtime inference session to use.</param>
    /// <param name="runOptions">The runtime options for execution.</param>
    /// <param name="options">The configuration options for the model executor.</param>
    /// <returns>A new instance of the executor of type <typeparamref name="T"/>.</returns>
    static abstract T Create(InferenceSession session, RunOptions runOptions, OnnxModelExecutorOptions options);
}

/// <summary>
/// Provides a base implementation for ONNX model executors.
/// </summary>
public abstract class OnnxModelExecutorBase : IModelExecutor<long, float>
{
    /// <summary>
    /// The ONNX runtime inference session used by this executor.
    /// </summary>
    protected readonly InferenceSession Session;

    /// <summary>
    /// The runtime options used by this executor.
    /// </summary>
    protected readonly RunOptions RunOptions;

    private readonly ConcurrentBag<long[]> _dimensionsPool;
    private readonly ConcurrentBag<Memory<long>[]> _inputMemoryPool;
    private readonly SemaphoreSlim? _semaphore;

    /// <summary>
    /// Initializes a new instance of the <see cref="OnnxModelExecutorBase"/> class.
    /// </summary>
    /// <param name="session">The ONNX runtime inference session to use.</param>
    /// <param name="runOptions">The runtime options for execution.</param>
    /// <param name="maxThreads">The maximum number of threads allowed, or <c>null</c> for no limit.</param>
    protected OnnxModelExecutorBase(InferenceSession session, RunOptions runOptions, int? maxThreads = null)
    {
        Session = session;
        RunOptions = runOptions;
        _dimensionsPool = [];
        _inputMemoryPool = [];
        _semaphore = maxThreads.HasValue ? new SemaphoreSlim(maxThreads.Value, maxThreads.Value) : null;
    }

    /// <summary>
    /// Runs inference using the ONNX runtime session with the provided inputs and tensor values.
    /// </summary>
    /// <param name="inputs">The input tensors for the model.</param>
    /// <param name="ortValues">The prepared ONNX tensor values.</param>
    /// <returns>A task representing the asynchronous inference operation, containing the result as a disposable collection of <see cref="OrtValue"/>.</returns>
    protected abstract Task<IDisposableReadOnlyCollection<OrtValue>> RunSessionInference(Tensor<long>[] inputs, OrtValue[] ortValues);

    /// <summary>
    /// Executes the model asynchronously with the provided input tensors.
    /// </summary>
    /// <param name="inputs">The input tensors for the model.</param>
    /// <returns>A task representing the asynchronous execution, containing the result as a disposable collection of <see cref="OrtValue"/>.</returns>
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

    /// <summary>
    /// Runs the model asynchronously and returns the output as an array of tensors.
    /// </summary>
    /// <param name="inputs">The input tensors for the model.</param>
    /// <returns>A task containing the output tensors.</returns>
    public async Task<Tensor<float>[]> RunAsync(Tensor<long>[] inputs)
    {
        using IDisposableReadOnlyCollection<OrtValue> result = await ExecuteModelAsync(inputs);
        return ToOutTensors(result);
    }

    /// <summary>
    /// Runs the model asynchronously and processes the output using the provided callback.
    /// </summary>
    /// <param name="inputs">The input tensors for the model.</param>
    /// <param name="postProcess">A callback to process each output tensor span.</param>
    /// <returns>A task representing the asynchronous operation.</returns>
    public async Task RunAsync(Tensor<long>[] inputs, Action<ReadOnlyTensorSpan<float>, int> postProcess)
    {
        using IDisposableReadOnlyCollection<OrtValue> result = await ExecuteModelAsync(inputs);
        for (int i = 0; i < result.Count; i++)
        {
            postProcess(result[i].GetTensorDataAsTensorSpan<float>(), i);
        }
    }

    /// <summary>
    /// Prepares input tensors as ONNX runtime tensor values.
    /// </summary>
    /// <param name="inputs">The input tensors for the model.</param>
    /// <returns>An array of prepared <see cref="OrtValue"/> tensors.</returns>
    protected virtual OrtValue[] GetModelInputs(Tensor<long>[] inputs)
    {
        long[] dims = GetInputDims(inputs);
        Memory<long>[] modelInputs = GetInputsAsMemory(inputs);
        OrtValue[] ortValues = modelInputs.AsSpan().ToOrtValues(dims);

        // Return to pool:
        _dimensionsPool.Add(dims);
        _inputMemoryPool.Add(modelInputs);

        return ortValues;
    }

    private static Tensor<float>[] ToOutTensors(IReadOnlyCollection<OrtValue> result)
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

        Tensor<float> outTensor = Tensor.CreateFromShape<float>(x.Lengths, x.Strides);
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