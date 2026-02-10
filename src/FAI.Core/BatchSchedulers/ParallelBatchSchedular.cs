using FAI.Core.Abstractions;
using FAI.Core.Configurations.PipelineBatchExecutors;

namespace FAI.Core.BatchSchedulers;

public class ParallelBatchSchedular<TIn, TOut> : IBatchSchedular<TIn, TOut>
{
    private readonly ParallelOptions _parallelOptions;

    public ParallelBatchSchedular(ParallelBatchSchedularOptions options)
    {
        _parallelOptions = options.MaxConcurrency.HasValue ? new ParallelOptions { MaxDegreeOfParallelism = options.MaxConcurrency.Value } : new ParallelOptions();
    }

    public async Task RunInExecutor(IPipelineBatchExecutor<TIn, TOut> executor, IEnumerable<Range> ranges, ReadOnlyMemory<TIn> inputs, Memory<TOut> outputs)
    {
        await Parallel.ForEachAsync(ranges, _parallelOptions, async (range, _) => await executor.ExecuteBatchPredict(inputs[range], outputs[range]));
    }
}
