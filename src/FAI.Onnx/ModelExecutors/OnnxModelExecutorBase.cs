using System.Collections.Concurrent;
using System.Numerics.Tensors;
using FAI.Core;
using FAI.Core.Steps;
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
    IAllocatingStep<Tensor<long>[], Tensor<float>[]>,
    IBorrowedTensorProducer<Tensor<long>[], float>
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

    public ValueTask<BatchLease<Tensor<float>[]>> RentOutputAsync(
        Tensor<long>[] input,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (input.Length == 0)
        {
            throw new ArgumentException("At least one model input tensor is required.", nameof(input));
        }

        Tensor<float>[] output = new Tensor<float>[Session.OutputNames.Count];
        for (int outputIndex = 0; outputIndex < output.Length; outputIndex++)
        {
            int[] dimensions = Session.OutputMetadata[Session.OutputNames[outputIndex]].Dimensions;
            var resolvedDimensions = new nint[dimensions.Length];
            for (int dimensionIndex = 0; dimensionIndex < dimensions.Length; dimensionIndex++)
            {
                int dimension = dimensions[dimensionIndex];
                if (dimension > 0)
                {
                    resolvedDimensions[dimensionIndex] = dimension;
                }
                else if (dimensionIndex < input[0].Rank)
                {
                    resolvedDimensions[dimensionIndex] = input[0].Lengths[dimensionIndex];
                }
                else
                {
                    throw new InvalidOperationException(
                        $"Output '{Session.OutputNames[outputIndex]}' dimension {dimensionIndex} is symbolic and cannot be inferred from the model input.");
                }
            }

            output[outputIndex] = Tensor.CreateFromShape<float>(resolvedDimensions);
        }

        return ValueTask.FromResult(new BatchLease<Tensor<float>[]>(output));
    }

    public async ValueTask ExecuteAsync(
        Tensor<long>[] input,
        Tensor<float>[] output,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (output.Length != Session.OutputNames.Count)
        {
            throw new ArgumentException(
                $"The model produces {Session.OutputNames.Count} outputs, but {output.Length} destinations were supplied.",
                nameof(output));
        }

        using IDisposableReadOnlyCollection<OrtValue> result = await ExecuteModelAsync(input);
        for (int outputIndex = 0; outputIndex < result.Count; outputIndex++)
        {
            ReadOnlyTensorSpan<float> outputData = result[outputIndex].GetTensorDataAsTensorSpan<float>();
            if (!outputData.Lengths.SequenceEqual(output[outputIndex].Lengths))
            {
                throw new ArgumentException(
                    $"Output {outputIndex} has shape [{string.Join(", ", output[outputIndex].Lengths.ToArray())}], " +
                    $"but the model produced [{string.Join(", ", outputData.Lengths.ToArray())}].",
                    nameof(output));
            }

            outputData.CopyTo(output[outputIndex]);
        }
        cancellationToken.ThrowIfCancellationRequested();
    }

    public async ValueTask ExecuteAsync<TOutput>(
        Tensor<long>[] input,
        TOutput output,
        IBorrowedTensorConsumer<float, TOutput> consumer,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        using IDisposableReadOnlyCollection<OrtValue> result = await ExecuteModelAsync(input);
        for (int outputIndex = 0; outputIndex < result.Count; outputIndex++)
        {
            consumer.Consume(result[outputIndex].GetTensorDataAsTensorSpan<float>(), outputIndex, output);
        }

        cancellationToken.ThrowIfCancellationRequested();
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
