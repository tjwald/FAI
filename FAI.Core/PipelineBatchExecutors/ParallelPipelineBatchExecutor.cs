using FAI.Core.Abstractions;

namespace FAI.Core.PipelineBatchExecutors;

/// <summary>
/// An batch executor that divides the input data into smaller chunks and processes them concurrently.
/// </summary>
/// <typeparam name="TInput">The type of the input data for batch processing.</typeparam>
/// <typeparam name="TOutput">The type of the output data for batch processing.</typeparam>
public class ParallelPipelineBatchExecutor<TInput, TOutput> : IPipelineBatchExecutor<TInput, TOutput>
{
    private readonly int _maxBatchSize;
    private readonly int? _maxConcurrency;
    private readonly IInferenceSteps<TInput, TOutput> _inferenceSteps;

    /// <summary>
    /// Initializes a new instance of the <see cref="ParallelPipelineBatchExecutor{TInput, TOutput}"/> class.
    /// </summary>
    /// <param name="inferenceSteps">The steps to process each batch of inputs.</param>
    /// <param name="maxBatchSize">The maximum size of each batch to process.</param>
    /// <param name="maxConcurrency">The maximum degree of parallelism. If null, no limit is applied.</param>
    public ParallelPipelineBatchExecutor(IInferenceSteps<TInput, TOutput> inferenceSteps, int maxBatchSize, int? maxConcurrency)
    {
        _inferenceSteps = inferenceSteps;
        _maxBatchSize = maxBatchSize;
        _maxConcurrency = maxConcurrency;
    }

    /// <summary>
    /// Processes input data in parallel by dividing it into batches and executing them concurrently.
    /// The results are written to the provided output memory block.
    /// </summary>
    /// <param name="inputs">The input data to process, provided as a read-only memory block.</param>
    /// <param name="outputSpan">The memory block where the processed output data will be written.</param>
    /// <returns>A task that represents the asynchronous operation, ensuring non-blocking execution.</returns>
    public async Task ExecuteBatchPredict(ReadOnlyMemory<TInput> inputs, Memory<TOutput> outputSpan)
    {
        int maxBatchSize = _maxBatchSize;
        int batchCount = inputs.Length / maxBatchSize;

        var parallelOptions = _maxConcurrency.HasValue ? new ParallelOptions { MaxDegreeOfParallelism = _maxConcurrency.Value } : new ParallelOptions();

        var task = Parallel.ForAsync(0, batchCount, parallelOptions, async (i, _) =>
        {
            int batchStartIndex = i * maxBatchSize;
            int batchEndIndex = batchStartIndex + maxBatchSize;
            await _inferenceSteps.ProcessBatch(inputs[batchStartIndex..batchEndIndex], outputSpan[batchStartIndex..batchEndIndex]);
        });

        if (inputs.Length % maxBatchSize > 0)
        {
            int batchStartIndex = batchCount * maxBatchSize;
            int batchEndIndex = inputs.Length;
            await _inferenceSteps.ProcessBatch(inputs[batchStartIndex..batchEndIndex], outputSpan[batchStartIndex..batchEndIndex]);
        }

        await task;
    }
}
