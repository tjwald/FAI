// ReSharper disable once CheckNamespace
namespace FAI.Core.Abstractions;

/// <summary>
/// Defines application-level inference operations.
/// </summary>
/// <typeparam name="TInput">The input type.</typeparam>
/// <typeparam name="TOutput">The output type.</typeparam>
public interface IBatchInference<TInput, TBatchOutput>
{
    /// <summary>
    /// Predicts outputs for a batch of inputs.
    /// </summary>
    Task<TBatchOutput> BatchPredict(ReadOnlyMemory<TInput> input);
}

/// <summary>
/// Defines application-level inference operations with one output per input.
/// </summary>
public interface IInference<TInput, TOutput> : IBatchInference<TInput, TOutput[]>
{
    /// <summary>
    /// Predicts one output for one input.
    /// </summary>
    Task<TOutput> Predict(TInput input);

    /// <summary>
    /// Predicts outputs into a caller-provided buffer.
    /// </summary>
    Task BatchPredict(ReadOnlyMemory<TInput> input, Memory<TOutput> output);
}
