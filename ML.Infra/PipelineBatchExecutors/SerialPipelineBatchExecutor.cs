using ML.Infra.Abstractions;

namespace ML.Infra.PipelineBatchExecutors;

public class SerialPipelineBatchExecutor<TInput, TOutput> : IPipelineBatchExecutor<TInput, TOutput>
{
    private readonly IInferenceSteps<TInput, TOutput> _inferenceSteps;
    private readonly int _maxBatchSize;

    public SerialPipelineBatchExecutor(IInferenceSteps<TInput, TOutput> inferenceSteps, int maxBatchSize)
    {
        _inferenceSteps = inferenceSteps;
        _maxBatchSize = maxBatchSize;
    }
    
    public async Task ExecuteBatchPredict(ReadOnlyMemory<TInput> inputs, Memory<TOutput> outputSpan)
    {
        int batchStartIndex = 0;
        for (; batchStartIndex < inputs.Length - _maxBatchSize; batchStartIndex += _maxBatchSize)
        {
            int batchEndIndex = batchStartIndex + _maxBatchSize;
            await _inferenceSteps.ProcessBatch(inputs[batchStartIndex..batchEndIndex], outputSpan[batchStartIndex..batchEndIndex]);
        }

        if (batchStartIndex < inputs.Length)
        {
            await _inferenceSteps.ProcessBatch(inputs[batchStartIndex..], outputSpan[batchStartIndex..]);
        }
    }
}