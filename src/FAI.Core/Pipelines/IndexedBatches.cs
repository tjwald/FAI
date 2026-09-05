using System.Buffers;
using System.Runtime.CompilerServices;

namespace FAI.Core.Pipelines;

public interface IIndexedBatchRegistry
{
    IWritableIndexedBatch<TBatch> GetWritable<TBatch>();

    IReadOnlyIndexedBatch<TBatch> GetReadOnly<TBatch>();
}

public sealed class IndexedBatchRegistry : IIndexedBatchRegistry
{
    private readonly IServiceProvider _serviceProvider;

    public IndexedBatchRegistry(IServiceProvider serviceProvider)
    {
        _serviceProvider = serviceProvider;
    }

    public IWritableIndexedBatch<TBatch> GetWritable<TBatch>()
    {
        var batch = (IWritableIndexedBatch<TBatch>?)_serviceProvider.GetService(typeof(IWritableIndexedBatch<TBatch>));
        if (batch is not null)
        {
            return batch;
        }

        throw new InvalidOperationException(
            $"No writable indexed batch operations are registered in the service collection for '{typeof(TBatch).Name}'. " +
            $"Register them using services.AddBatchOperations<{typeof(TBatch).Name}, ...>() or services.AddMemoryBatch<T>().");
    }

    public IReadOnlyIndexedBatch<TBatch> GetReadOnly<TBatch>()
    {
        var batch = (IReadOnlyIndexedBatch<TBatch>?)_serviceProvider.GetService(typeof(IReadOnlyIndexedBatch<TBatch>));
        if (batch is not null)
        {
            return batch;
        }

        throw new InvalidOperationException(
            $"No read-only indexed batch operations are registered in the service collection for '{typeof(TBatch).Name}'. " +
            $"Register them using services.AddBatchOperations<{typeof(TBatch).Name}, ...>() or services.AddReadOnlyMemoryBatch<T>().");
    }
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

    public void Copy(Memory<T> source, Memory<T> destination)
        => source.Span.CopyTo(destination.Span);

    public void Scatter(Memory<T> source, Memory<T> destination, ReadOnlySpan<int> destinationIndices)
    {
        ReadOnlySpan<T> sourceSpan = source.Span;
        Span<T> destinationSpan = destination.Span;
        for (int i = 0; i < destinationIndices.Length; i++)
        {
            destinationSpan[destinationIndices[i]] = sourceSpan[i];
        }
    }

    public void PermuteInPlace(Memory<T> batch, Span<int> sourceToDestinationIndices)
    {
        Span<T> values = batch.Span;
        for (int sourceIndex = 0; sourceIndex < sourceToDestinationIndices.Length; sourceIndex++)
        {
            int destinationIndex = sourceToDestinationIndices[sourceIndex];
            if (destinationIndex == sourceIndex)
            {
                continue;
            }

            T value = values[sourceIndex];
            while (destinationIndex != sourceIndex)
            {
                (value, values[destinationIndex]) = (values[destinationIndex], value);
                int nextDestinationIndex = sourceToDestinationIndices[destinationIndex];
                sourceToDestinationIndices[destinationIndex] = destinationIndex;
                destinationIndex = nextDestinationIndex;
            }

            values[sourceIndex] = value;
            sourceToDestinationIndices[sourceIndex] = sourceIndex;
        }
    }
}
