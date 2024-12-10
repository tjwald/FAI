using ML.Infra.Abstractions;

namespace ML.Infra.PipelineBatchExecutors;

public readonly struct ParallelPipelineBatchExecutor<TInput, TOutput, TPreprocess, TModelOutput> : IPipelineBatchExecutor<TInput, TOutput, TPreprocess, TModelOutput>
{
    private readonly int _maxBatchSize;
    private readonly int _maxConcurrency;

    public ParallelPipelineBatchExecutor(int maxBatchSize, int? maxConcurrency)
    {
        _maxBatchSize = maxBatchSize;
        _maxConcurrency = maxConcurrency ?? 0;
    }
    
    public async Task ExecuteBatchPredict(Pipeline<TInput, TOutput, TPreprocess, TModelOutput> pipeline, ReadOnlyMemory<TInput> inputs, Memory<TOutput> outputSpan)
    {
        int maxBatchSize = _maxBatchSize;
        int batchCount = inputs.Length / maxBatchSize;
        if (inputs.Length % maxBatchSize != 0)
        {
            batchCount++;
        }
        
        await Parallel.ForAsync(0, batchCount, new ParallelOptions {MaxDegreeOfParallelism = _maxConcurrency}, async (i, _) =>
        {
            int batchStartIndex = i * maxBatchSize;
            int batchEndIndex = batchStartIndex + maxBatchSize;
            if (batchEndIndex > inputs.Length)
            {
                batchEndIndex = inputs.Length;
            }
            await pipeline.ProcessBatch(inputs[batchStartIndex..batchEndIndex], outputSpan[batchStartIndex..batchEndIndex]);
        });
    }
}