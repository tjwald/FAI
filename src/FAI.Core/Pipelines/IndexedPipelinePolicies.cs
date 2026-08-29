namespace FAI.Core.Pipelines;

public interface IBatchPartitioner<in TBatch>
{
    IEnumerable<Range> Partition(TBatch batch);
}

public interface IIndexOrdering<in TBatch>
{
    int[] CreateOrder(TBatch batch);
}

public sealed class PartitioningPipeline<TInput, TOutput>
    : IPreallocatingPipeline<TInput, TOutput>
{
    private readonly IPipeline<TInput, TOutput> _inner;
    private readonly IPreallocatingPipeline<TInput, TOutput>? _preallocatingInner;
    private readonly IBatchPartitioner<TInput> _partitioner;
    private readonly IPartitionScheduler _scheduler;
    private readonly IReadOnlyIndexedBatch<TInput> _inputBatch;
    private readonly IWritableIndexedBatch<TOutput> _outputBatch;

    public PartitioningPipeline(
        IPipeline<TInput, TOutput> inner,
        IBatchPartitioner<TInput> partitioner,
        IReadOnlyIndexedBatch<TInput> inputBatch,
        IWritableIndexedBatch<TOutput> outputBatch,
        IPartitionScheduler? scheduler = null)
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
            try
            {
                await ExecuteAsync(input, output, cancellationToken);
                return output;
            }
            catch
            {
                await PipelineOutputDisposer.DisposeAsync(output);
                throw;
            }
        }

        Range[] ranges = _partitioner.Partition(input).ToArray();
        if (ranges.Length == 0)
        {
            throw new InvalidOperationException("Partitioning an empty batch requires preallocation support.");
        }

        var partitionOutputs = new TOutput[ranges.Length];
        Dictionary<Range, int> rangeIndices = ranges
            .Select((range, index) => (range, index))
            .ToDictionary(item => item.range, item => item.index);
        try
        {
            await _scheduler.ExecuteAsync(
                ranges,
                async (range, token) =>
                {
                    int index = rangeIndices[range];
                    partitionOutputs[index] = await _inner.ExecuteAsync(
                        _inputBatch.Slice(input, range),
                        token);
                },
                cancellationToken);

            TOutput aggregate = _outputBatch.AllocateLike(partitionOutputs[0], _inputBatch.Count(input));
            try
            {
                for (int i = 0; i < ranges.Length; i++)
                {
                    ScatterRange(partitionOutputs[i], aggregate, ranges[i]);
                }

                return aggregate;
            }
            catch
            {
                await PipelineOutputDisposer.DisposeAsync(aggregate);
                throw;
            }
        }
        finally
        {
            foreach (TOutput partitionOutput in partitionOutputs)
            {
                if (partitionOutput is not null)
                {
                    await PipelineOutputDisposer.DisposeAsync(partitionOutput);
                }
            }
        }
    }

    public async ValueTask ExecuteAsync(
        TInput input,
        TOutput output,
        CancellationToken cancellationToken = default)
    {
        int inputCount = _inputBatch.Count(input);
        if (inputCount != _outputBatch.Count(output))
        {
            throw new ArgumentException("Input and output batch counts must match.", nameof(output));
        }

        await _scheduler.ExecuteAsync(
            _partitioner.Partition(input),
            async (range, token) =>
            {
                TInput partitionInput = _inputBatch.Slice(input, range);
                if (_preallocatingInner is not null)
                {
                    await _preallocatingInner.ExecuteAsync(
                        partitionInput,
                        _outputBatch.Slice(output, range),
                        token);
                    return;
                }

                TOutput partitionOutput = await _inner.ExecuteAsync(partitionInput, token);
                try
                {
                    ScatterRange(partitionOutput, output, range);
                }
                finally
                {
                    await PipelineOutputDisposer.DisposeAsync(partitionOutput);
                }
            },
            cancellationToken);
    }

    private void ScatterRange(TOutput source, TOutput destination, Range range)
    {
        (int offset, int length) = range.GetOffsetAndLength(_outputBatch.Count(destination));
        int[] destinationIndices = Enumerable.Range(offset, length).ToArray();
        _outputBatch.Scatter(source, destination, destinationIndices);
    }

}

public sealed class OrderingPipeline<TInput, TOutput>
    : IPreallocatingPipeline<TInput, TOutput>
{
    private readonly IPipeline<TInput, TOutput> _inner;
    private readonly IPreallocatingPipeline<TInput, TOutput>? _preallocatingInner;
    private readonly IIndexOrdering<TInput> _ordering;
    private readonly IReadOnlyIndexedBatch<TInput> _inputBatch;
    private readonly IWritableIndexedBatch<TOutput> _outputBatch;

    public OrderingPipeline(
        IPipeline<TInput, TOutput> inner,
        IIndexOrdering<TInput> ordering,
        IReadOnlyIndexedBatch<TInput> inputBatch,
        IWritableIndexedBatch<TOutput> outputBatch)
    {
        _inner = inner;
        _preallocatingInner = inner as IPreallocatingPipeline<TInput, TOutput>;
        _ordering = ordering;
        _inputBatch = inputBatch;
        _outputBatch = outputBatch;
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
        int[] sortedToOriginal = _ordering.CreateOrder(input);
        using BatchLease<TInput> sortedInput = _inputBatch.Gather(input, sortedToOriginal);
        TOutput output = await _inner.ExecuteAsync(sortedInput.Value, cancellationToken);
        _outputBatch.PermuteInPlace(output, sortedToOriginal);
        return output;
    }

    public async ValueTask ExecuteAsync(
        TInput input,
        TOutput output,
        CancellationToken cancellationToken = default)
    {
        int[] sortedToOriginal = _ordering.CreateOrder(input);
        using BatchLease<TInput> sortedInput = _inputBatch.Gather(input, sortedToOriginal);

        if (_preallocatingInner is not null)
        {
            await _preallocatingInner.ExecuteAsync(sortedInput.Value, output, cancellationToken);
        }
        else
        {
            TOutput sortedOutput = await _inner.ExecuteAsync(sortedInput.Value, cancellationToken);
            try
            {
                int[] identity = Enumerable.Range(0, _outputBatch.Count(sortedOutput)).ToArray();
                _outputBatch.Scatter(sortedOutput, output, identity);
            }
            finally
            {
                await PipelineOutputDisposer.DisposeAsync(sortedOutput);
            }
        }

        _outputBatch.PermuteInPlace(output, sortedToOriginal);
    }
}
