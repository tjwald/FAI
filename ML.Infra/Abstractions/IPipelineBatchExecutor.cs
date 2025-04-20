namespace ML.Infra.Abstractions;

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
