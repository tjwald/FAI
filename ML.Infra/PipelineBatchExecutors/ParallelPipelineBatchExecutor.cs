using ML.Infra.Abstractions;

namespace ML.Infra.PipelineBatchExecutors;

public readonly struct ParallelPipelineBatchExecutor<TInput, TOutput> : IPipelineBatchExecutor<TInput, TOutput>
{
    private readonly int _maxBatchSize;
    private readonly int? _maxConcurrency;

    public ParallelPipelineBatchExecutor(int maxBatchSize, int? maxConcurrency)
    {
        _maxBatchSize = maxBatchSize;
        _maxConcurrency = maxConcurrency;
    }
    
    public async Task ExecuteBatchPredict(IPipeline<TInput, TOutput> pipeline, ReadOnlyMemory<TInput> inputs, Memory<TOutput> outputSpan)
    {
        int maxBatchSize = _maxBatchSize;
        int batchCount = inputs.Length / maxBatchSize;

        var parallelOptions = _maxConcurrency.HasValue ? new ParallelOptions { MaxDegreeOfParallelism = _maxConcurrency.Value } : new ParallelOptions();
        
        var task = Parallel.ForAsync(0, batchCount, parallelOptions, async (i, _) =>
        {
            int batchStartIndex = i * maxBatchSize;
            int batchEndIndex = batchStartIndex + maxBatchSize;
            await pipeline.ProcessBatch(inputs[batchStartIndex..batchEndIndex], outputSpan[batchStartIndex..batchEndIndex]);
        });

        if (inputs.Length % maxBatchSize > 0)
        {
            int batchStartIndex = batchCount * maxBatchSize;
            int batchEndIndex = inputs.Length;
            await pipeline.ProcessBatch(inputs[batchStartIndex..batchEndIndex], outputSpan[batchStartIndex..batchEndIndex]);
        }

        await task;
    }
}