using System.Collections.Concurrent;
using System.Numerics.Tensors;
using FAI.Core;
using FAI.Core.Pipelines;
using FAI.Core.Utilities;
using FAI.Onnx.Configuration;
using FAI.Onnx.Utils;
using Microsoft.ML.OnnxRuntime;

namespace FAI.Onnx.ModelExecutors;

/// <summary>
/// Defines a contract for an ONNX model executor.
/// </summary>
/// <typeparam name="T">The type of the executor implementing this interface.</typeparam>
public interface IOnnxModelExecutor<out T> where T : IOnnxModelExecutor<T>
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
public abstract class OnnxModelExecutorBase :
    IPipeline<Tensor<long>[], TensorOutputs<float>>,
    IPipeline<BatchEncode, TensorOutputs<float>>
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
    /// <param name="cancellationToken">The cancellation token to observe.</param>
    /// <returns>A task representing the asynchronous inference operation, containing the result as a disposable collection of <see cref="OrtValue"/>.</returns>
    protected abstract Task<IDisposableReadOnlyCollection<OrtValue>> RunSessionInference(
        IReadOnlyList<string> inputNames,
        OrtValue[] ortValues,
        int batchSize,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Executes the model asynchronously with the provided input tensors.
    /// </summary>
    /// <param name="inputs">The input tensors for the model.</param>
    /// <param name="cancellationToken">The cancellation token to observe.</param>
    /// <returns>A task representing the asynchronous execution, containing the result as a disposable collection of <see cref="OrtValue"/>.</returns>
    private async Task<IDisposableReadOnlyCollection<OrtValue>> ExecuteModelAsync(
        BatchEncode inputs,
        CancellationToken cancellationToken)
    {
        return await ExecuteModelAsyncCore(
            () => GetModelInputs(inputs),
            checked((int)inputs.InputIds.Lengths[0]),
            cancellationToken);
    }

    private async Task<IDisposableReadOnlyCollection<OrtValue>> ExecuteModelAsync(
        Tensor<long>[] inputs,
        CancellationToken cancellationToken)
    {
        return await ExecuteModelAsyncCore(
            () => (ResolveInputNamesOrThrow(inputs.Length), GetModelInputs(inputs)),
            checked((int)inputs[0].Lengths[0]),
            cancellationToken);
    }

    private async Task<IDisposableReadOnlyCollection<OrtValue>> ExecuteModelAsyncCore(
        Func<(string[] InputNames, OrtValue[] OrtValues)> prepareInputs,
        int batchSize,
        CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        using (await _semaphore.EnterScope(cancellationToken))
        {
            cancellationToken.ThrowIfCancellationRequested();
            (string[] inputNames, OrtValue[] ortValues) = prepareInputs();
            try
            {
                return await RunSessionInference(inputNames, ortValues, batchSize, cancellationToken);
            }
            finally
            {
                foreach (OrtValue input in ortValues)
                {
                    input.Dispose();
                }
            }
        }
    }

    public async ValueTask<TensorOutputs<float>> ExecuteAsync(
        BatchEncode input,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (input.InputIds is null)
        {
            throw new ArgumentException("BatchEncode requires an input_ids tensor.", nameof(input));
        }

        IDisposableReadOnlyCollection<OrtValue> result = await ExecuteModelAsync(input, cancellationToken);
        try
        {
            cancellationToken.ThrowIfCancellationRequested();
            return new OnnxTensorOutputs(result);
        }
        catch
        {
            result.Dispose();
            throw;
        }
    }

    public ValueTask<TensorOutputs<float>> ExecuteAsync(
        Tensor<long>[] input,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (input.Length == 0)
        {
            throw new ArgumentException("At least one model input tensor is required.", nameof(input));
        }

        return ExecuteAsyncLegacy(input, cancellationToken);
    }

    private async ValueTask<TensorOutputs<float>> ExecuteAsyncLegacy(
        Tensor<long>[] input,
        CancellationToken cancellationToken)
    {
        IDisposableReadOnlyCollection<OrtValue> result = await ExecuteModelAsync(input, cancellationToken);
        try
        {
            cancellationToken.ThrowIfCancellationRequested();
            return new OnnxTensorOutputs(result);
        }
        catch
        {
            result.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Prepares input tensors as ONNX runtime tensor values.
    /// </summary>
    /// <param name="inputs">The input tensors for the model.</param>
    /// <returns>An array of prepared <see cref="OrtValue"/> tensors.</returns>
    protected virtual (string[] InputNames, OrtValue[] OrtValues) GetModelInputs(BatchEncode inputs)
    {
        (string[] inputNames, Tensor<long>[] tensorInputs) = ResolveModelInputTensors(inputs);
        OrtValue[] ortValues = GetModelInputs(tensorInputs);
        return (inputNames, ortValues);
    }

    protected virtual OrtValue[] GetModelInputs(Tensor<long>[] inputs)
    {
        long[] dims = GetInputDims(inputs);
        Memory<long>[] modelInputs = GetInputsAsMemory(inputs);
        OrtValue[] ortValues = modelInputs.AsSpan().ToOrtValues(dims);

        _dimensionsPool.Add(dims);
        _inputMemoryPool.Add(modelInputs);

        return ortValues;
    }

    protected string[] ResolveInputNamesForBatchEncode(BatchEncode inputs)
    {
        return ResolveModelInputTensors(inputs).InputNames;
    }

    private (string[] InputNames, Tensor<long>[] Tensors) ResolveModelInputTensors(BatchEncode inputs)
    {
        List<string> inputNames = [];
        List<Tensor<long>> tensors = [];
        foreach (string modelInputName in Session.InputNames)
        {
            switch (modelInputName)
            {
                case BatchEncode.InputIdsName:
                    inputNames.Add(modelInputName);
                    tensors.Add(inputs.InputIds);
                    break;
                case BatchEncode.AttentionMaskName when inputs.AttentionMask is not null:
                    inputNames.Add(modelInputName);
                    tensors.Add(inputs.AttentionMask);
                    break;
                case BatchEncode.AttentionMaskName:
                    throw new InvalidOperationException("Model requires attention_mask, but BatchEncode did not include it.");
                case BatchEncode.TokenTypeIdsName when inputs.TokenTypeIds is not null:
                    inputNames.Add(modelInputName);
                    tensors.Add(inputs.TokenTypeIds);
                    break;
                case BatchEncode.TokenTypeIdsName:
                    throw new InvalidOperationException("Model requires token_type_ids, but BatchEncode did not include it.");
            }
        }

        if (inputNames.Count > 0)
        {
            return ([.. inputNames], [.. tensors]);
        }

        if (inputs.Count == 1)
        {
            return (ResolveInputNamesOrThrow(1), [inputs.InputIds]);
        }

        throw new InvalidOperationException(
            $"Model does not expose Hugging Face input names ({BatchEncode.InputIdsName}, {BatchEncode.AttentionMaskName}, {BatchEncode.TokenTypeIdsName}) required for multi-tensor BatchEncode.");
    }

    private string[] ResolveInputNamesOrThrow(int inputCount)
    {
        if (Session.InputNames.Count < inputCount)
        {
            throw new InvalidOperationException($"Model expects {Session.InputNames.Count} inputs, but {inputCount} were provided.");
        }

        return Session.InputNames.Take(inputCount).ToArray();
    }

    private Memory<long>[] GetInputsAsMemory(Tensor<long>[] inputs)
    {
        Memory<long>[] modelInputs;
        if (!_inputMemoryPool.TryTake(out modelInputs!) || modelInputs.Length != inputs.Length)
        {
            modelInputs = new Memory<long>[inputs.Length];
        }

        for (int i = 0; i < inputs.Length; i++)
        {
            modelInputs[i] = inputs[i].AsMemory();
        }

        return modelInputs;
    }

    private long[] GetInputDims(Tensor<long>[] inputs)
    {
        long[] dims;
        if (!_dimensionsPool.TryTake(out dims!) || dims.Length != inputs[0].Rank)
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

internal sealed class OnnxTensorOutputs(IDisposableReadOnlyCollection<OrtValue> outputs) : TensorOutputs<float>
{
    public override int Count => outputs.Count;

    public override ReadOnlyTensorSpan<float> GetOutput(int index)
        => outputs[index].GetTensorDataAsTensorSpan<float>();

    public override void Dispose() => outputs.Dispose();
}
