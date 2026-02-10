using FAI.Core.Abstractions;

namespace FAI.Core.BatchSchedulers;

public sealed class ParallelBatchSchedularOptions
{
    public int MaxParallelism { get; init; } = Environment.ProcessorCount;
}

public class ParallelBatchSchedular<TIn, TOut> : IBatchSchedular<TIn, TOut>
{
    private readonly ParallelOptions _parallelOptions;

    public ParallelBatchSchedular(ParallelBatchSchedularOptions options)
    {
        _parallelOptions = new ParallelOptions { MaxDegreeOfParallelism = options.MaxParallelism };
    }

    public async Task RunInExecutor(IPipelineBatchExecutor<TIn, TOut> executor, IEnumerable<Range> ranges, ReadOnlyMemory<TIn> inputs, Memory<TOut> outputs)
    {
        await Parallel.ForEachAsync(ranges, _parallelOptions, async (range, _) => await executor.ExecuteBatchPredict(inputs[range], outputs[range]));
    }
}
