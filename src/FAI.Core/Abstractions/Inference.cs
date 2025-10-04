namespace FAI.Core.Abstractions;

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
