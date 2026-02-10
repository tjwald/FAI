using System.Numerics.Tensors;

// ReSharper disable once CheckNamespace
namespace FAI.Core.Abstractions;

/// <summary>
/// Represents a contract for executing machine learning models with specified input and output types.
/// </summary>
/// <typeparam name="TInput">The type of the input data for the model.</typeparam>
/// <typeparam name="TOutput">The type of the output data from the model.</typeparam>
public interface IModelExecutor<TInput, TOutput>
{
    /// <summary>
    /// Executes the model asynchronously with the given inputs and returns the outputs as an array of tensors.
    /// </summary>
    /// <param name="inputs">An array of input tensors to be processed by the model.</param>
    /// <returns>A task that represents the asynchronous operation. The task result contains an array of output tensors.</returns>
    Task<Tensor<TOutput>[]> RunAsync(Tensor<TInput>[] inputs);

    /// <summary>
    /// Executes the model asynchronously with the given inputs and processes the outputs using a provided callback function.
    /// </summary>
    /// <param name="inputs">An array of input tensors to be processed by the model.</param>
    /// <param name="postProcess">
    /// A callback function to process the output tensors. The function receives a read-only span of the output tensor data
    /// and the index of the corresponding input tensor.
    /// </param>
    /// <returns>A task that represents the asynchronous operation.</returns>
    Task RunAsync(Tensor<TInput>[] inputs, Action<ReadOnlyTensorSpan<TOutput>, int> postProcess);
}

public interface IPreprocessor<TInput, out TPreprocessContainer, TFloat> where TPreprocessContainer : IEnumerable<Tensor<TFloat>>
{
    TPreprocessContainer Preprocess(ReadOnlySpan<TInput> input);
}

public interface IBatchSchedular<TIn, TOut>
{
    Task RunInExecutor(IPipelineBatchExecutor<TIn, TOut> executor, IEnumerable<Range> ranges, ReadOnlyMemory<TIn> inputs, Memory<TOut> outputs);
}

public interface IBatchSlicer<TIn>
{
    IEnumerable<Range> Slice(ReadOnlyMemory<TIn> inputs);
}

public interface IFailedBatchPolicy<TInput, TOutput>
{
    /// <summary>
    /// Handles a batch failure. The policy can choose to:
    /// 1. Retry execution using the provided 'executor'.
    /// 2. Log the error and rethrow (default).
    /// 3. Fill 'outputs' with fallback values and return (suppress error).
    /// </summary>
    Task HandleAsync(
        ReadOnlyMemory<TInput> inputs,
        Memory<TOutput> outputs,
        IPipelineBatchExecutor<TInput, TOutput> executor,
        Exception originalException,
        CancellationToken ct);
}

/// <summary>
/// Represents a contract for executing batch predictions in a machine learning pipeline.
/// This interface enables efficient processing of multiple inputs in a single operation,
/// supporting asynchronous execution and memory-efficient data handling.
/// </summary>
/// <typeparam name="TInput">The type of the input data for the batch prediction.</typeparam>
/// <typeparam name="TOutput">The type of the output data for the batch prediction.</typeparam>
public interface IPipelineBatchExecutor<TInput, TOutput>
{
    /// <summary>
    /// Executes a batch prediction operation, processing a collection of inputs
    /// and writing the corresponding outputs to the provided memory block.
    /// </summary>
    /// <param name="inputs">The input data for the batch prediction, provided as a read-only memory block.</param>
    /// <param name="outputSpan">The memory block where the output data will be written.</param>
    /// <returns>A task that represents the asynchronous operation, ensuring non-blocking execution.</returns>
    Task ExecuteBatchPredict(ReadOnlyMemory<TInput> inputs, Memory<TOutput> outputSpan);
}

/// <summary>
/// Represents a pipeline that processes input of type <typeparamref name="TInput"/>
/// and produces output of type <typeparamref name="TOutput"/>.
/// </summary>
/// <typeparam name="TInput">The type of the input data.</typeparam>
/// <typeparam name="TOutput">The type of the output data.</typeparam>
public interface IPipeline<TInput, TOutput> : IInference<TInput, TOutput>;

/// <summary>
/// Defines the contract for inference steps that process input data and produce output data.
/// </summary>
/// <typeparam name="TInput">The type of the input data.</typeparam>
/// <typeparam name="TOutput">The type of the output data.</typeparam>
public interface IInferenceSteps<TInput, TOutput>
{
    /// <summary>
    /// Processes a batch of input data and produces corresponding output data.
    /// </summary>
    /// <param name="inputs">The input data as a read-only memory block.</param>
    /// <param name="outputs">The output data as a writable memory block.</param>
    /// <returns>A task that represents the asynchronous operation.</returns>
    Task ProcessBatch(ReadOnlyMemory<TInput> inputs, Memory<TOutput> outputs);
}

/// <summary>
/// Provides an abstract base class for implementing inference steps with preprocessing, model execution, and postprocessing.
/// </summary>
/// <typeparam name="TInput">The type of the input data.</typeparam>
/// <typeparam name="TPreprocess">The type of the preprocessing result.</typeparam>
/// <typeparam name="TModelOutput">The type of the model output.</typeparam>
/// <typeparam name="TOutput">The type of the final output data.</typeparam>
public abstract class InferenceSteps<TInput, TPreprocess, TModelOutput, TOutput> : IInferenceSteps<TInput, TOutput>
{
    /// <summary>
    /// Preprocesses the input data to prepare it for model execution.
    /// </summary>
    /// <param name="input">The input data as a read-only span.</param>
    /// <returns>The result of preprocessing.</returns>
    public abstract TPreprocess Preprocess(ReadOnlySpan<TInput> input);

    /// <summary>
    /// Executes the model using the input data and preprocessing results.
    /// </summary>
    /// <param name="input">The input data as a read-only memory block.</param>
    /// <param name="preprocesses">The preprocessing results.</param>
    /// <returns>A task that represents the asynchronous operation, containing the model output.</returns>
    public abstract Task<TModelOutput> RunModel(ReadOnlyMemory<TInput> input, TPreprocess preprocesses);

    /// <summary>
    /// Postprocesses the model output to produce the final output data.
    /// </summary>
    /// <param name="inputs">The input data as a read-only span.</param>
    /// <param name="preprocesses">The preprocessing results.</param>
    /// <param name="modelOutput">The model output.</param>
    /// <param name="outputs">The final output data as a writable span.</param>
    public abstract void PostProcess(ReadOnlySpan<TInput> inputs, TPreprocess preprocesses, TModelOutput modelOutput, Span<TOutput> outputs);

    /// <summary>
    /// Processes a batch of input data by performing preprocessing, model execution, and postprocessing.
    /// </summary>
    /// <param name="inputs">The input data as a read-only memory block.</param>
    /// <param name="outputs">The output data as a writable memory block.</param>
    /// <returns>A task that represents the asynchronous operation.</returns>
    public async Task ProcessBatch(ReadOnlyMemory<TInput> inputs, Memory<TOutput> outputs)
    {
        var preprocess = Preprocess(inputs.Span);
        var modelOutput = await RunModel(inputs, preprocess);
        PostProcess(inputs.Span, preprocess, modelOutput, outputs.Span);
    }
}

/// <summary>
/// Defines an interface for performing inference operations.
/// </summary>
/// <typeparam name="TInput">The type of the input data for the inference.</typeparam>
/// <typeparam name="TOutput">The type of the output data from the inference.</typeparam>
public interface IInference<TInput, TOutput>
{
    /// <summary>
    /// Predicts the output based on a single input.
    /// </summary>
    /// <param name="input">The input data for the prediction.</param>
    Task<TOutput> Predict(TInput input);

    /// <summary>
    /// Predicts the outputs for a batch of inputs.
    /// </summary>
    /// <param name="input">A read-only memory containing the batch of input data for the predictions.</param>
    Task<TOutput[]> BatchPredict(ReadOnlyMemory<TInput> input);

    /// <summary>
    /// Predicts the outputs for a batch of inputs.
    /// </summary>
    /// <param name="input">A read-only memory containing the batch of input data for the predictions.</param>
    /// <param name="output">An output buffer for the results</param>
    Task BatchPredict(ReadOnlyMemory<TInput> input, Memory<TOutput> output);
}
