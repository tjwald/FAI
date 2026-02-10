using FAI.Core.Abstractions;

namespace FAI.Core.PipelineBatchExecutors;

public sealed class PartitionPipelineBatchExecutor<TIn, TOut> : IPipelineBatchExecutor<TIn, TOut>
{
    private readonly IBatchSchedular<TIn, TOut> _schedular;
    private readonly IBatchSlicer<TIn> _batchSlicer;
    private readonly IPipelineBatchExecutor<TIn, TOut> _executor;

    public PartitionPipelineBatchExecutor(IBatchSchedular<TIn, TOut> schedular, IBatchSlicer<TIn> batchSlicer, IPipelineBatchExecutor<TIn, TOut> executor)
    {
        _schedular = schedular;
        _batchSlicer = batchSlicer;
        _executor = executor;
    }

    public async Task ExecuteBatchPredict(ReadOnlyMemory<TIn> inputs, Memory<TOut> outputSpan)
    {
        var ranges = _batchSlicer.Slice(inputs);

        await _schedular.RunInExecutor(_executor, ranges, inputs, outputSpan);
    }
}
