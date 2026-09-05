namespace FAI.Core.Pipelines;

public sealed class PartitioningPipeline<TInput, TOutput> : IDestinationPipeline<TInput, TOutput>
{
    private readonly IDestinationPipeline<TInput, TOutput> _inner;
    private readonly IBatchPartitioner<TInput> _partitioner;
    private readonly IPartitionScheduler _scheduler;
    private readonly IReadOnlyIndexedBatch<TInput> _inputBatch;
    private readonly IWritableIndexedBatch<TOutput> _outputBatch;

    public PartitioningPipeline(IPipeline<TInput, TOutput> inner, IBatchPartitioner<TInput> partitioner,
        IReadOnlyIndexedBatch<TInput> inputBatch, IWritableIndexedBatch<TOutput> outputBatch, IPartitionScheduler? scheduler = null)
    {
        ArgumentNullException.ThrowIfNull(inner);
        ArgumentNullException.ThrowIfNull(partitioner);
        ArgumentNullException.ThrowIfNull(inputBatch);
        ArgumentNullException.ThrowIfNull(outputBatch);

        _inner = inner.AsDestinationPipeline(outputBatch);
        _partitioner = partitioner;
        _inputBatch = inputBatch;
        _outputBatch = outputBatch;
        _scheduler = scheduler ?? new SerialPartitionScheduler();
    }

    public async ValueTask<TOutput> ExecuteAsync(TInput input, CancellationToken cancellationToken = default)
    {
        if (_inputBatch.Count(input) == 0)
        {
            return await _inner.ExecuteAsync(input, cancellationToken);
        }

        Range[] ranges = [.. _partitioner.Partition(input)];
        if (ranges.Length == 0)
        {
            return await _inner.ExecuteAsync(input, cancellationToken);
        }

        var partitionOutputs = new TOutput[ranges.Length];
        try
        {
            await _scheduler.ExecuteAsync(ranges, async (range, token) =>
            {
                int index = FindRangeIndex(ranges, range);
                partitionOutputs[index] = await _inner.ExecuteAsync(_inputBatch.Slice(input, range), token);
            }, cancellationToken);

            TOutput aggregate = _outputBatch.AllocateLike(partitionOutputs[0], _inputBatch.Count(input));
            try
            {
                for (int i = 0; i < ranges.Length; i++)
                {
                    _outputBatch.Copy(partitionOutputs[i], _outputBatch.Slice(aggregate, ranges[i]));
                }

                return aggregate;
            }
            catch { await PipelineOutputDisposer.DisposeAsync(aggregate); throw; }
        }
        finally
        {
            foreach (TOutput partitionOutput in partitionOutputs)
                if (partitionOutput is not null) await PipelineOutputDisposer.DisposeAsync(partitionOutput);
        }
    }

    public async ValueTask ExecuteAsync(TInput input, TOutput destination, CancellationToken cancellationToken = default)
    {
        if (_inputBatch.Count(input) != _outputBatch.Count(destination))
            throw new ArgumentException("Input and output batch counts must match.", nameof(destination));

        await _scheduler.ExecuteAsync(_partitioner.Partition(input), async (range, token) =>
        {
            TInput partitionInput = _inputBatch.Slice(input, range);
            TOutput partitionDestination = _outputBatch.Slice(destination, range);
            await _inner.ExecuteAsync(partitionInput, partitionDestination, token);
        }, cancellationToken);
    }

    private static int FindRangeIndex(ReadOnlySpan<Range> ranges, Range range)
    {
        for (int index = 0; index < ranges.Length; index++) if (ranges[index].Equals(range)) return index;
        throw new InvalidOperationException("The scheduler returned an unknown partition range.");
    }
}
