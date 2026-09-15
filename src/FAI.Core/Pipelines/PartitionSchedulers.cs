using FAI.Core.Configurations;

namespace FAI.Core.Pipelines;

public sealed class SerialPartitionScheduler : IPartitionScheduler
{
    public async ValueTask ExecuteAsync(
        IEnumerable<Range> ranges,
        Func<Range, CancellationToken, ValueTask> execute,
        CancellationToken cancellationToken = default)
    {
        foreach (Range range in ranges)
        {
            cancellationToken.ThrowIfCancellationRequested();
            await execute(range, cancellationToken);
        }
    }
}

public sealed class ParallelPartitionScheduler : IPartitionScheduler
{
    private readonly int? _maxConcurrency;

    public ParallelPartitionScheduler(ParallelPartitionSchedulerOptions options)
    {
        _maxConcurrency = options.MaxConcurrency;
    }

    public async ValueTask ExecuteAsync(
        IEnumerable<Range> ranges,
        Func<Range, CancellationToken, ValueTask> execute,
        CancellationToken cancellationToken = default)
    {
        var parallelOptions = new ParallelOptions
        {
            CancellationToken = cancellationToken,
            MaxDegreeOfParallelism = _maxConcurrency ?? -1,
        };
        await Parallel.ForEachAsync(
            ranges,
            parallelOptions,
            async (range, token) => await execute(range, token));
    }
}
