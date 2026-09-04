namespace FAI.Core.Pipelines;

public sealed class OrderingPipeline<TInput, TOutput> : IDestinationPipeline<TInput, TOutput>
{
    private readonly IPipeline<TInput, TOutput> _inner;
    private readonly IDestinationPipeline<TInput, TOutput>? _destinationInner;
    private readonly IIndexOrdering<TInput> _ordering;
    private readonly IReadOnlyIndexedBatch<TInput> _inputBatch;
    private readonly IWritableIndexedBatch<TOutput> _outputBatch;

    public OrderingPipeline(IPipeline<TInput, TOutput> inner, IIndexOrdering<TInput> ordering,
        IReadOnlyIndexedBatch<TInput> inputBatch, IWritableIndexedBatch<TOutput> outputBatch)
    {
        _inner = inner;
        _destinationInner = inner as IDestinationPipeline<TInput, TOutput>;
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
        if (_destinationInner is not null)
        {
            await _destinationInner.ExecuteAsync(sortedInput.Value, destination, cancellationToken);
        }
        else
        {
            TOutput sortedOutput = await _inner.ExecuteAsync(sortedInput.Value, cancellationToken);
            try
            {
                int outputCount = _outputBatch.Count(sortedOutput);
                Span<int> identity = stackalloc int[outputCount];
                for (int index = 0; index < outputCount; index++) identity[index] = index;
                _outputBatch.Scatter(sortedOutput, destination, identity);
            }
            finally { await PipelineOutputDisposer.DisposeAsync(sortedOutput); }
        }

        _outputBatch.PermuteInPlace(destination, sortedToOriginal);
    }
}
