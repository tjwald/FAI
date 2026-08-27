using System.Buffers;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;

namespace FAI.Core.Steps;

public sealed class TensorBatchOperations<T> :
    IReadOnlyIndexedBatch<Tensor<T>, TensorBatchOperations<T>>,
    IWritableIndexedBatch<Tensor<T>, TensorBatchOperations<T>>
{
    public static int Count(Tensor<T> batch) => checked((int)batch.Lengths[0]);

    public static Tensor<T> Slice(Tensor<T> batch, Range range)
    {
        (int offset, int length) = range.GetOffsetAndLength(Count(batch));
        var ranges = new NRange[batch.Rank];
        ranges[0] = new NRange(offset, offset + length);
        for (int dimension = 1; dimension < ranges.Length; dimension++)
        {
            ranges[dimension] = NRange.All;
        }

        return batch.Slice(ranges);
    }

    public static BatchLease<Tensor<T>> Gather(Tensor<T> source, ReadOnlySpan<int> indices)
    {
        BatchLease<Tensor<T>> lease = RentLike(source, indices.Length);
        Tensor<T> destination = lease.Value;
        for (int destinationIndex = 0; destinationIndex < indices.Length; destinationIndex++)
        {
            CopyRow(source, indices[destinationIndex], destination, destinationIndex);
        }

        return lease;
    }

    public static BatchLease<Tensor<T>> RentLike(Tensor<T> template, int count)
    {
        nint[] lengths = template.Lengths.ToArray();
        lengths[0] = count;
        int elementCount = GetElementCount(lengths);
        T[] buffer = ArrayPool<T>.Shared.Rent(elementCount);
        Tensor<T> tensor = Tensor.Create(buffer, 0, lengths, template.Strides);

        return new BatchLease<Tensor<T>>(
            tensor,
            _ => ArrayPool<T>.Shared.Return(buffer, RuntimeHelpers.IsReferenceOrContainsReferences<T>()));
    }

    public static void Scatter(Tensor<T> source, Tensor<T> destination, ReadOnlySpan<int> destinationIndices)
    {
        if (Count(source) != destinationIndices.Length)
        {
            throw new ArgumentException("A destination index is required for every source row.", nameof(destinationIndices));
        }

        for (int sourceIndex = 0; sourceIndex < destinationIndices.Length; sourceIndex++)
        {
            CopyRow(source, sourceIndex, destination, destinationIndices[sourceIndex]);
        }
    }

    public static void PermuteInPlace(Tensor<T> batch, Span<int> sourceToDestinationIndices)
    {
        using BatchLease<Tensor<T>> copy = Gather(batch, Enumerable.Range(0, Count(batch)).ToArray());
        Scatter(copy.Value, batch, sourceToDestinationIndices);
    }

    private static void CopyRow(Tensor<T> source, int sourceIndex, Tensor<T> destination, int destinationIndex)
    {
        Tensor<T> sourceRow = Slice(source, sourceIndex..(sourceIndex + 1));
        Tensor<T> destinationRow = Slice(destination, destinationIndex..(destinationIndex + 1));
        sourceRow.AsReadOnlyTensorSpan().CopyTo(destinationRow.AsTensorSpan());
    }

    private static int GetElementCount(ReadOnlySpan<nint> lengths)
    {
        int count = 1;
        foreach (nint length in lengths)
        {
            count = checked(count * (int)length);
        }

        return count;
    }
}
