using System.Numerics.Tensors;

namespace FAI.Core.Pipelines;

/// <summary>
/// A composable pipeline step that preallocates a destination tensor using captured n-1 trailing dimensions
/// and the input batch dimension, enabling zero-allocation destination execution across downstream stages.
/// </summary>
/// <typeparam name="TInput">The input type.</typeparam>
/// <typeparam name="TElement">The tensor element type.</typeparam>
public sealed class TensorDestinationAllocationPipeline<TInput, TElement> : IDestinationPipeline<TInput, Tensor<TElement>>
{
    private readonly IDestinationPipeline<TInput, Tensor<TElement>> _inner;
    private readonly IReadOnlyIndexedBatch<TInput> _inputBatch;
    private nint[]? _trailingDimensions;

    /// <summary>
    /// Initializes a new instance of the <see cref="TensorDestinationAllocationPipeline{TInput, TElement}"/> class.
    /// </summary>
    public TensorDestinationAllocationPipeline(
        IPipeline<TInput, Tensor<TElement>> inner,
        IReadOnlyIndexedBatch<TInput> inputBatch,
        IWritableIndexedBatch<Tensor<TElement>> outputBatch,
        ReadOnlySpan<nint> trailingDimensions = default)
    {
        ArgumentNullException.ThrowIfNull(inner);
        ArgumentNullException.ThrowIfNull(inputBatch);
        ArgumentNullException.ThrowIfNull(outputBatch);

        _inner = inner.AsDestinationPipeline(outputBatch);
        _inputBatch = inputBatch;
        if (!trailingDimensions.IsEmpty)
        {
            _trailingDimensions = trailingDimensions.ToArray();
        }
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="TensorDestinationAllocationPipeline{TInput, TElement}"/> class.
    /// </summary>
    public TensorDestinationAllocationPipeline(
        IPipeline<TInput, Tensor<TElement>> inner,
        IReadOnlyIndexedBatch<TInput> inputBatch,
        IWritableIndexedBatch<Tensor<TElement>> outputBatch,
        params int[] trailingDimensions)
        : this(inner, inputBatch, outputBatch, Array.ConvertAll(trailingDimensions, x => (nint)x))
    {
    }

    /// <inheritdoc />
    public async ValueTask<Tensor<TElement>> ExecuteAsync(TInput input, CancellationToken cancellationToken = default)
    {
        int batchSize = _inputBatch.Count(input);
        if (_trailingDimensions is { Length: > 0 } trailing)
        {
            nint[] shape = new nint[trailing.Length + 1];
            shape[0] = batchSize;
            trailing.CopyTo(shape.AsSpan(1));
            Tensor<TElement> destination = Tensor.CreateFromShape<TElement>(shape);
            await _inner.ExecuteAsync(input, destination, cancellationToken);
            return destination;
        }

        Tensor<TElement> output = await _inner.ExecuteAsync(input, cancellationToken);
        if (output.Rank > 1)
        {
            _trailingDimensions = output.Lengths.Slice(1).ToArray();
        }

        return output;
    }

    /// <inheritdoc />
    public ValueTask ExecuteAsync(TInput input, Tensor<TElement> destination, CancellationToken cancellationToken = default)
    {
        if (_trailingDimensions is null && destination.Rank > 1)
        {
            _trailingDimensions = destination.Lengths.Slice(1).ToArray();
        }

        return _inner.ExecuteAsync(input, destination, cancellationToken);
    }
}
