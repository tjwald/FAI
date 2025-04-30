using ML.Infra.Abstractions;

namespace ML.Infra.PipelineBatchExecutors;

/// <summary>
/// Pipeline executor that processes batches in sequence.
/// </summary>
/// <typeparam name="TInput">The type of the input data for the batch prediction.</typeparam>
/// <typeparam name="TOutput">The type of the output data for the batch prediction.</typeparam>
public class SerialPipelineBatchExecutor<TInput, TOutput> : IPipelineBatchExecutor<TInput, TOutput>
{
    private readonly IInferenceSteps<TInput, TOutput> _inferenceSteps;
    private readonly int? _maxBatchSize;

    /// <summary>
    /// Initializes a new instance of the <see cref="SerialPipelineBatchExecutor{TInput, TOutput}"/> class.
    /// </summary>
    /// <param name="inferenceSteps">The inference steps to process the input data and produce output data.</param>
    /// <param name="maxBatchSize">The maximum size of each batch to be processed.</param>
    public SerialPipelineBatchExecutor(IInferenceSteps<TInput, TOutput> inferenceSteps, int? maxBatchSize)
    {
        _inferenceSteps = inferenceSteps;
        _maxBatchSize = maxBatchSize;
    }

    /// <summary>
    /// Executes a batch prediction operation by processing the input data in sequential batches.
    /// The results are written to the provided output memory block.
    /// </summary>
    /// <param name="inputs">The input data for the batch prediction, provided as a read-only memory block.</param>
    /// <param name="outputSpan">The memory block where the output data will be written.</param>
    /// <returns>A task that represents the asynchronous operation, ensuring non-blocking execution.</returns>
    public async Task ExecuteBatchPredict(ReadOnlyMemory<TInput> inputs, Memory<TOutput> outputSpan)
    {
        if (!_maxBatchSize.HasValue)
        {
            await _inferenceSteps.ProcessBatch(inputs, outputSpan);
            return;
        }
        
        int maxBatchSize = _maxBatchSize.Value;
        int batchStartIndex = 0;
        for (; batchStartIndex < inputs.Length - maxBatchSize; batchStartIndex += maxBatchSize)
        {
            int batchEndIndex = batchStartIndex + maxBatchSize;
            await _inferenceSteps.ProcessBatch(inputs[batchStartIndex..batchEndIndex], outputSpan[batchStartIndex..batchEndIndex]);
        }

        if (batchStartIndex < inputs.Length)
        {
            await _inferenceSteps.ProcessBatch(inputs[batchStartIndex..], outputSpan[batchStartIndex..]);
        }
    }
}
