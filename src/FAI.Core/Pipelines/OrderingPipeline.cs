namespace FAI.Core.Pipelines;

public sealed class OrderingPipeline<TInput, TOutput> : IDestinationPipeline<TInput, TOutput>
{
    private readonly IDestinationPipeline<TInput, TOutput> _inner;
    private readonly IIndexOrdering<TInput> _ordering;
    private readonly IReadOnlyIndexedBatch<TInput> _inputBatch;
    private readonly IWritableIndexedBatch<TOutput> _outputBatch;

    public OrderingPipeline(IPipeline<TInput, TOutput> inner, IIndexOrdering<TInput> ordering,
        IReadOnlyIndexedBatch<TInput> inputBatch, IWritableIndexedBatch<TOutput> outputBatch)
    {
        ArgumentNullException.ThrowIfNull(inner);
        ArgumentNullException.ThrowIfNull(ordering);
        ArgumentNullException.ThrowIfNull(inputBatch);
        ArgumentNullException.ThrowIfNull(outputBatch);

        _inner = inner.AsDestinationPipeline(outputBatch);
        _ordering = ordering;
        _inputBatch = inputBatch;
        _outputBatch = outputBatch;
    }

    public async ValueTask<TOutput> ExecuteAsync(TInput input, CancellationToken cancellationToken = default)
    {
        int[] sortedToOriginal = _ordering.CreateOrder(input);
        using BatchLease<TInput> sortedInput = _inputBatch.Gather(input, sortedToOriginal);
        TOutput output = await _inner.ExecuteAsync(sortedInput.Value, cancellationToken);
        _outputBatch.PermuteInPlace(output, sortedToOriginal);
        return output;
    }

    public async ValueTask ExecuteAsync(TInput input, TOutput destination, CancellationToken cancellationToken = default)
    {
        int[] sortedToOriginal = _ordering.CreateOrder(input);
        using BatchLease<TInput> sortedInput = _inputBatch.Gather(input, sortedToOriginal);
        await _inner.ExecuteAsync(sortedInput.Value, destination, cancellationToken);
        _outputBatch.PermuteInPlace(destination, sortedToOriginal);
    }
}
