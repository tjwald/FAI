using System.Numerics;
using System.Numerics.Tensors;

namespace FAI.Core.Steps;

public interface IBorrowedTensorConsumer<TElement, in TOutput>
    where TElement : unmanaged, INumber<TElement>
{
    void Consume(ReadOnlyTensorSpan<TElement> tensor, int outputIndex, TOutput output);
}

public interface IBorrowedTensorProducer<in TInput, TElement>
    where TElement : unmanaged, INumber<TElement>
{
    ValueTask ExecuteAsync<TOutput>(
        TInput input,
        TOutput output,
        IBorrowedTensorConsumer<TElement, TOutput> consumer,
        CancellationToken cancellationToken = default);
}

public sealed class BorrowedTensorDecodingStep<TInput, TElement, TOutput> : IAllocatingStep<TInput, TOutput>
    where TElement : unmanaged, INumber<TElement>
{
    private readonly IBorrowedTensorProducer<TInput, TElement> _producer;
    private readonly IBorrowedTensorConsumer<TElement, TOutput> _consumer;
    private readonly Func<TInput, CancellationToken, ValueTask<BatchLease<TOutput>>> _rentOutput;

    public BorrowedTensorDecodingStep(
        IBorrowedTensorProducer<TInput, TElement> producer,
        IBorrowedTensorConsumer<TElement, TOutput> consumer,
        Func<TInput, CancellationToken, ValueTask<BatchLease<TOutput>>> rentOutput)
    {
        _producer = producer;
        _consumer = consumer;
        _rentOutput = rentOutput;
    }

    public ValueTask<BatchLease<TOutput>> RentOutputAsync(
        TInput input,
        CancellationToken cancellationToken = default)
        => _rentOutput(input, cancellationToken);

    public async ValueTask<BatchLease<TOutput>> ExecuteAsync(
        TInput input,
        CancellationToken cancellationToken = default)
    {
        BatchLease<TOutput> output = await _rentOutput(input, cancellationToken);
        try
        {
            await ExecuteAsync(input, output.Value, cancellationToken);
            return output;
        }
        catch
        {
            output.Dispose();
            throw;
        }
    }

    public ValueTask ExecuteAsync(
        TInput input,
        TOutput output,
        CancellationToken cancellationToken = default)
        => _producer.ExecuteAsync(input, output, _consumer, cancellationToken);
}
