namespace FAI.Core.Pipelines;

public sealed class PartitioningPipeline<TInput, TOutput> : IPreallocatingPipeline<TInput, TOutput>
{
    private readonly IPipeline<TInput, TOutput> _inner;
    private readonly IPreallocatingPipeline<TInput, TOutput>? _preallocatingInner;
    private readonly IBatchPartitioner<TInput> _partitioner;
    private readonly IPartitionScheduler _scheduler;
    private readonly IReadOnlyIndexedBatch<TInput> _inputBatch;
    private readonly IWritableIndexedBatch<TOutput> _outputBatch;

    public PartitioningPipeline(IPipeline<TInput, TOutput> inner, IBatchPartitioner<TInput> partitioner,
        IReadOnlyIndexedBatch<TInput> inputBatch, IWritableIndexedBatch<TOutput> outputBatch, IPartitionScheduler? scheduler = null)
    {
        _inner = inner;
        _preallocatingInner = inner as IPreallocatingPipeline<TInput, TOutput>;
        _partitioner = partitioner;
        _inputBatch = inputBatch;
        _outputBatch = outputBatch;
        _scheduler = scheduler ?? new SerialPartitionScheduler();
    }

    public bool TryAllocateOutput(TInput input, out TOutput output)
    {
        if (_preallocatingInner is not null && _preallocatingInner.TryAllocateOutput(input, out TOutput? allocated))
        {
            output = allocated;
            return true;
        }

        output = default!;
        return false;
    }

    public async ValueTask<TOutput> ExecuteAsync(TInput input, CancellationToken cancellationToken = default)
    {
        if (TryAllocateOutput(input, out TOutput? output))
        {
            try { await ExecuteAsync(input, output, cancellationToken); return output; }
            catch { await PipelineOutputDisposer.DisposeAsync(output); throw; }
        }

        Range[] ranges = [.. _partitioner.Partition(input)];
        if (ranges.Length == 0) throw new InvalidOperationException("Partitioning an empty batch requires preallocation support.");

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
                for (int i = 0; i < ranges.Length; i++) ScatterRange(partitionOutputs[i], aggregate, ranges[i]);
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

    public async ValueTask ExecuteAsync(TInput input, TOutput output, CancellationToken cancellationToken = default)
    {
        if (_inputBatch.Count(input) != _outputBatch.Count(output))
            throw new ArgumentException("Input and output batch counts must match.", nameof(output));

        await _scheduler.ExecuteAsync(_partitioner.Partition(input), async (range, token) =>
        {
            TInput partitionInput = _inputBatch.Slice(input, range);
            if (_preallocatingInner is not null)
            {
                await _preallocatingInner.ExecuteAsync(partitionInput, _outputBatch.Slice(output, range), token);
                return;
            }

            TOutput partitionOutput = await _inner.ExecuteAsync(partitionInput, token);
            try { ScatterRange(partitionOutput, output, range); }
            finally { await PipelineOutputDisposer.DisposeAsync(partitionOutput); }
        }, cancellationToken);
    }

    private void ScatterRange(TOutput source, TOutput destination, Range range)
    {
        (int offset, int length) = range.GetOffsetAndLength(_outputBatch.Count(destination));
        Span<int> destinationIndices = stackalloc int[length];
        for (int index = 0; index < length; index++) destinationIndices[index] = offset + index;
        _outputBatch.Scatter(source, destination, destinationIndices);
    }

    private static int FindRangeIndex(ReadOnlySpan<Range> ranges, Range range)
    {
        for (int index = 0; index < ranges.Length; index++) if (ranges[index].Equals(range)) return index;
        throw new InvalidOperationException("The scheduler returned an unknown partition range.");
    }
}
