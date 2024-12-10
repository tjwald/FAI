using ML.Infra.Abstractions;

namespace ML.Infra.PipelineBatchExecutors;

public readonly struct SerialPipelineBatchExecutor<TInput, TOutput, TPreprocess, TModelOutput> : IPipelineBatchExecutor<TInput, TOutput, TPreprocess, TModelOutput>
{
    private readonly int _maxBatchSize;

    public SerialPipelineBatchExecutor(int maxBatchSize)
    {
        _maxBatchSize = maxBatchSize;
    }
    
    public async Task ExecuteBatchPredict(Pipeline<TInput, TOutput, TPreprocess, TModelOutput> pipeline, ReadOnlyMemory<TInput> inputs, Memory<TOutput> outputSpan)
    {
        int batchStartIndex = 0;
        for (; batchStartIndex < inputs.Length - _maxBatchSize; batchStartIndex += _maxBatchSize)
        {
            int batchEndIndex = batchStartIndex + _maxBatchSize;
            await pipeline.ProcessBatch(inputs[batchStartIndex..batchEndIndex], outputSpan[batchStartIndex..batchEndIndex]);
        }

        if (batchStartIndex < inputs.Length)
        {
            await pipeline.ProcessBatch(inputs[batchStartIndex..], outputSpan[batchStartIndex..]);
        }
    }
}