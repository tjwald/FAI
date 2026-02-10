using FAI.Core.Abstractions;

namespace FAI.Core.BatchSchedulers;

public class SerialBatchSchedular<TIn, TOut> : IBatchSchedular<TIn, TOut>
{
    public async Task RunInExecutor(IPipelineBatchExecutor<TIn, TOut> executor, IEnumerable<Range> ranges, ReadOnlyMemory<TIn> inputs, Memory<TOut> outputs)
    {
        foreach (Range range in ranges)
        {
            await executor.ExecuteBatchPredict(inputs[range], outputs[range]);
        }
    }
}
