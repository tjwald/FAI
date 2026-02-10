using FAI.Core.Abstractions;

namespace FAI.Core.PipelineBatchExecutors;

public sealed class SinkPipelineBatchExecutor<TInput, TOutput> : IPipelineBatchExecutor<TInput, TOutput>
{
    private readonly IInferenceSteps<TInput, TOutput> _inferenceSteps;

    public SinkPipelineBatchExecutor(IInferenceSteps<TInput, TOutput> inferenceSteps)
    {
        _inferenceSteps = inferenceSteps;
    }

    public async Task ExecuteBatchPredict(ReadOnlyMemory<TInput> inputs, Memory<TOutput> outputSpan)
    {
        await _inferenceSteps.ProcessBatch(inputs, outputSpan);
    }
}
