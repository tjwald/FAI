using FAI.Core.Configurations;

namespace FAI.Core.Pipelines;

public interface IPartitionScheduler
{
    ValueTask ExecuteAsync(
        IEnumerable<Range> ranges,
        Func<Range, CancellationToken, ValueTask> execute,
        CancellationToken cancellationToken = default);
}

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
    private readonly ParallelOptions _parallelOptions;

    public ParallelPartitionScheduler(ParallelPartitionSchedulerOptions options)
    {
        _parallelOptions = options.MaxConcurrency.HasValue
            ? new ParallelOptions { MaxDegreeOfParallelism = options.MaxConcurrency.Value }
            : new ParallelOptions();
    }

    public async ValueTask ExecuteAsync(
        IEnumerable<Range> ranges,
        Func<Range, CancellationToken, ValueTask> execute,
        CancellationToken cancellationToken = default)
    {
        _parallelOptions.CancellationToken = cancellationToken;
        await Parallel.ForEachAsync(
            ranges,
            _parallelOptions,
            async (range, token) => await execute(range, token));
    }
}
