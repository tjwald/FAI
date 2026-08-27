using System.Buffers;
using System.Runtime.CompilerServices;

namespace FAI.Core.Steps;

public interface IReadOnlyIndexedBatch<TBatch, TSelf>
    where TSelf : IReadOnlyIndexedBatch<TBatch, TSelf>
{
    static abstract int Count(TBatch batch);

    static abstract TBatch Slice(TBatch batch, Range range);

    static abstract BatchLease<TBatch> Gather(TBatch source, ReadOnlySpan<int> indices);
}

public interface IWritableIndexedBatch<TBatch, TSelf>
    where TSelf : IWritableIndexedBatch<TBatch, TSelf>
{
    static abstract int Count(TBatch batch);

    static abstract TBatch Slice(TBatch batch, Range range);

    static abstract BatchLease<TBatch> RentLike(TBatch template, int count);

    static abstract void Scatter(TBatch source, TBatch destination, ReadOnlySpan<int> destinationIndices);

    static abstract void PermuteInPlace(TBatch batch, Span<int> sourceToDestinationIndices);
}

public sealed class ReadOnlyMemoryBatchOperations<T> : IReadOnlyIndexedBatch<ReadOnlyMemory<T>, ReadOnlyMemoryBatchOperations<T>>
{
    public static int Count(ReadOnlyMemory<T> batch) => batch.Length;

    public static ReadOnlyMemory<T> Slice(ReadOnlyMemory<T> batch, Range range) => batch[range];

    public static BatchLease<ReadOnlyMemory<T>> Gather(ReadOnlyMemory<T> source, ReadOnlySpan<int> indices)
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

public sealed class MemoryBatchOperations<T> : IWritableIndexedBatch<Memory<T>, MemoryBatchOperations<T>>
{
    public static int Count(Memory<T> batch) => batch.Length;

    public static Memory<T> Slice(Memory<T> batch, Range range) => batch[range];

    public static BatchLease<Memory<T>> RentLike(Memory<T> template, int count)
    {
        T[] buffer = ArrayPool<T>.Shared.Rent(count);
        return new BatchLease<Memory<T>>(
            buffer.AsMemory(0, count),
            _ => ArrayPool<T>.Shared.Return(buffer, RuntimeHelpers.IsReferenceOrContainsReferences<T>()));
    }

    public static void Scatter(Memory<T> source, Memory<T> destination, ReadOnlySpan<int> destinationIndices)
    {
        ReadOnlySpan<T> sourceSpan = source.Span;
        Span<T> destinationSpan = destination.Span;
        for (int i = 0; i < destinationIndices.Length; i++)
        {
            destinationSpan[destinationIndices[i]] = sourceSpan[i];
        }
    }

    public static void PermuteInPlace(Memory<T> batch, Span<int> sourceToDestinationIndices)
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
