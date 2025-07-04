namespace FAI.Core.Abstractions;

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