using System.Buffers;
using System.Runtime.CompilerServices;

namespace FAI.Core.Pipelines;

public static class IndexedBatchOperations
{
    public static IWritableIndexedBatch<TBatch> GetWritable<TBatch>()
        => WritableOperations<TBatch>.Instance;

    private static class WritableOperations<TBatch>
    {
        public static readonly IWritableIndexedBatch<TBatch> Instance = Create();

        private static IWritableIndexedBatch<TBatch> Create()
        {
            Type batchType = typeof(TBatch);
            if (!batchType.IsGenericType)
            {
                throw UnsupportedBatchType(batchType);
            }

            Type genericType = batchType.GetGenericTypeDefinition();
            Type itemType = batchType.GetGenericArguments()[0];
            Type? operationsType = genericType == typeof(Memory<>)
                ? typeof(MemoryBatchOperations<>).MakeGenericType(itemType)
                : genericType == typeof(System.Numerics.Tensors.Tensor<>)
                    ? typeof(TensorBatchOperations<>).MakeGenericType(itemType)
                    : null;

            return operationsType is not null
                ? (IWritableIndexedBatch<TBatch>)Activator.CreateInstance(operationsType)!
                : throw UnsupportedBatchType(batchType);
        }
    }

    private static InvalidOperationException UnsupportedBatchType(Type batchType)
        => new($"No writable indexed batch operations are registered for '{batchType}'.");
}

public interface IReadOnlyIndexedBatch<TBatch>
{
    int Count(TBatch batch);

    TBatch Slice(TBatch batch, Range range);

    BatchLease<TBatch> Gather(TBatch source, ReadOnlySpan<int> indices);
}

public interface IWritableIndexedBatch<TBatch> : IReadOnlyIndexedBatch<TBatch>
{

    TBatch AllocateLike(TBatch template, int count);

    void Scatter(TBatch source, TBatch destination, ReadOnlySpan<int> destinationIndices);

    void PermuteInPlace(TBatch batch, ReadOnlySpan<int> sourceToDestinationIndices);
}

public sealed class ReadOnlyMemoryBatchOperations<T> : IReadOnlyIndexedBatch<ReadOnlyMemory<T>>
{
    public int Count(ReadOnlyMemory<T> batch) => batch.Length;

    public ReadOnlyMemory<T> Slice(ReadOnlyMemory<T> batch, Range range) => batch[range];

    public BatchLease<ReadOnlyMemory<T>> Gather(ReadOnlyMemory<T> source, ReadOnlySpan<int> indices)
    {
        T[] buffer = ArrayPool<T>.Shared.Rent(indices.Length);
        ReadOnlySpan<T> sourceSpan = source.Span;
        for (int i = 0; i < indices.Length; i++)
        {
            buffer[i] = sourceSpan[indices[i]];
        }

        return new BatchLease<ReadOnlyMemory<T>>(
            buffer.AsMemory(0, indices.Length),
            _ => ArrayPool<T>.Shared.Return(buffer, RuntimeHelpers.IsReferenceOrContainsReferences<T>()));
    }
}

public sealed class MemoryBatchOperations<T> : IWritableIndexedBatch<Memory<T>>
{
    public int Count(Memory<T> batch) => batch.Length;

    public Memory<T> Slice(Memory<T> batch, Range range) => batch[range];

    public BatchLease<Memory<T>> Gather(Memory<T> source, ReadOnlySpan<int> indices)
    {
        T[] buffer = ArrayPool<T>.Shared.Rent(indices.Length);
        ReadOnlySpan<T> sourceSpan = source.Span;
        for (int i = 0; i < indices.Length; i++)
        {
            buffer[i] = sourceSpan[indices[i]];
        }

        return new BatchLease<Memory<T>>(
            buffer.AsMemory(0, indices.Length),
            _ => ArrayPool<T>.Shared.Return(buffer, RuntimeHelpers.IsReferenceOrContainsReferences<T>()));
    }

    public Memory<T> AllocateLike(Memory<T> template, int count) => new T[count];

    public void Scatter(Memory<T> source, Memory<T> destination, ReadOnlySpan<int> destinationIndices)
    {
        ReadOnlySpan<T> sourceSpan = source.Span;
        Span<T> destinationSpan = destination.Span;
        for (int i = 0; i < destinationIndices.Length; i++)
        {
            destinationSpan[destinationIndices[i]] = sourceSpan[i];
        }
    }

    public void PermuteInPlace(Memory<T> batch, ReadOnlySpan<int> sourceToDestinationIndices)
    {
        T[] copy = ArrayPool<T>.Shared.Rent(batch.Length);
        try
        {
            batch.Span.CopyTo(copy);
            Span<T> destination = batch.Span;
            for (int sourceIndex = 0; sourceIndex < sourceToDestinationIndices.Length; sourceIndex++)
            {
                destination[sourceToDestinationIndices[sourceIndex]] = copy[sourceIndex];
            }
        }
        finally
        {
            ArrayPool<T>.Shared.Return(copy, RuntimeHelpers.IsReferenceOrContainsReferences<T>());
        }
    }
}
