namespace FAI.Core.Steps;

public interface IBatchPartitioner<in TBatch>
{
    IEnumerable<Range> Partition(TBatch batch);
}

public interface IIndexOrdering<in TBatch>
{
    int[] CreateOrder(TBatch batch);
}

public sealed class PartitioningStep<TInput, TOutput, TInputBatch, TOutputBatch>
    : IAllocatingStep<TInput, TOutput>
    where TInputBatch : IReadOnlyIndexedBatch<TInput, TInputBatch>
    where TOutputBatch : IWritableIndexedBatch<TOutput, TOutputBatch>
{
    private readonly IAllocatingStep<TInput, TOutput> _inner;
    private readonly IBatchPartitioner<TInput> _partitioner;
    private readonly IPartitionScheduler _scheduler;

    public PartitioningStep(
        IAllocatingStep<TInput, TOutput> inner,
        IBatchPartitioner<TInput> partitioner,
        IPartitionScheduler? scheduler = null)
    {
        _inner = inner;
        _partitioner = partitioner;
        _scheduler = scheduler ?? new SerialPartitionScheduler();
    }

    public ValueTask<BatchLease<TOutput>> RentOutputAsync(TInput input, CancellationToken cancellationToken = default)
        => _inner.RentOutputAsync(input, cancellationToken);

    public async ValueTask ExecuteAsync(TInput input, TOutput output, CancellationToken cancellationToken = default)
    {
        int inputCount = TInputBatch.Count(input);
        int outputCount = TOutputBatch.Count(output);
        if (inputCount != outputCount)
        {
            throw new ArgumentException($"Input count {inputCount} does not match output count {outputCount}.", nameof(output));
        }

        await _scheduler.ExecuteAsync(
            _partitioner.Partition(input),
            (range, token) => _inner.ExecuteAsync(
                TInputBatch.Slice(input, range),
                TOutputBatch.Slice(output, range),
                token),
            cancellationToken);
    }
}

public sealed class OrderingStep<TInput, TOutput, TInputBatch, TOutputBatch>
    : IAllocatingStep<TInput, TOutput>
    where TInputBatch : IReadOnlyIndexedBatch<TInput, TInputBatch>
    where TOutputBatch : IWritableIndexedBatch<TOutput, TOutputBatch>
{
    private readonly IAllocatingStep<TInput, TOutput> _inner;
    private readonly IIndexOrdering<TInput> _ordering;

    public OrderingStep(IAllocatingStep<TInput, TOutput> inner, IIndexOrdering<TInput> ordering)
    {
        _inner = inner;
        _ordering = ordering;
    }

    public ValueTask<BatchLease<TOutput>> RentOutputAsync(TInput input, CancellationToken cancellationToken = default)
        => _inner.RentOutputAsync(input, cancellationToken);

    public async ValueTask ExecuteAsync(TInput input, TOutput output, CancellationToken cancellationToken = default)
    {
        int[] sortedToOriginal = _ordering.CreateOrder(input);

        using BatchLease<TInput> sortedInput = TInputBatch.Gather(input, sortedToOriginal);
        await _inner.ExecuteAsync(sortedInput.Value, output, cancellationToken);
        TOutputBatch.PermuteInPlace(output, sortedToOriginal);
    }
}
