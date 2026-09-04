using System.Buffers;
using System.Numerics.Tensors;

namespace FAI.Core.Pipelines;

public sealed class TensorBatchOperations<T> :
    IReadOnlyIndexedBatch<Tensor<T>>,
    IWritableIndexedBatch<Tensor<T>>
{
    public int Count(Tensor<T> batch) => checked((int)batch.Lengths[0]);

    public Tensor<T> Slice(Tensor<T> batch, Range range)
    {
        (int offset, int length) = range.GetOffsetAndLength(Count(batch));

        Span<NRange> ranges = stackalloc NRange[batch.Rank];
        ranges[0] = new NRange(offset, offset + length);
        ranges[1..].Fill(NRange.All);

        return batch.Slice(ranges);
    }

    public BatchLease<Tensor<T>> Gather(Tensor<T> source, ReadOnlySpan<int> indices)
    {
        Span<nint> batchLengths = stackalloc nint[source.Lengths.Length];
        batchLengths[0] = indices.Length;
        source.Lengths[1..].CopyTo(batchLengths[1..]);

        T[] buffer = ArrayPool<T>.Shared.Rent(GetElementCount(batchLengths));
        Tensor<T> tensor = Tensor.Create(buffer, 0, batchLengths, source.Strides);
        var lease = new BatchLease<Tensor<T>>(tensor, _ => ArrayPool<T>.Shared.Return(buffer));
        Tensor<T> destination = lease.Value;

        var sourceEnumerator = source.GetDimensionSpan(0);
        var destinationEnumerator = destination.GetDimensionSpan(0);

        for (int destinationIndex = 0; destinationIndex < indices.Length; destinationIndex++)
        {
            var sourceRow = sourceEnumerator[indices[destinationIndex]];
            var destinationRow = destinationEnumerator[destinationIndex];
            sourceRow.CopyTo(destinationRow);
        }

        return lease;
    }

    public Tensor<T> AllocateLike(Tensor<T> template, int count)
    {
        nint[] lengths = template.Lengths.ToArray();
        lengths[0] = count;
        return Tensor.CreateFromShape<T>(lengths);
    }

    public void Scatter(Tensor<T> source, Tensor<T> destination, ReadOnlySpan<int> destinationIndices)
    {
        if (Count(source) != destinationIndices.Length)
        {
            throw new ArgumentException("A destination index is required for every source row.", nameof(destinationIndices));
        }

        var sourceEnumerator = source.GetDimensionSpan(0);
        var destinationEnumerator = destination.GetDimensionSpan(0);

        for (int sourceIndex = 0; sourceIndex < destinationIndices.Length; sourceIndex++)
        {
            var sourceRow = sourceEnumerator[sourceIndex];
            var destinationRow = destinationEnumerator[destinationIndices[sourceIndex]];
            sourceRow.CopyTo(destinationRow);
        }
    }

    public void PermuteInPlace(Tensor<T> batch, ReadOnlySpan<int> sourceToDestinationIndices)
    {
        if (sourceToDestinationIndices.Length == 0)
        {
            return;
        }

        var batchView = batch.GetDimensionSpan(0);
        var tempLengths = batchView[sourceToDestinationIndices[0]].Lengths;
        var flattenedLength = GetElementCount(tempLengths);
        T[] buffer = ArrayPool<T>.Shared.Rent(flattenedLength * 2);
        int[] permutation = ArrayPool<int>.Shared.Rent(sourceToDestinationIndices.Length);
        sourceToDestinationIndices.CopyTo(permutation);
        try
        {
            TensorSpan<T> currentHolding = new TensorSpan<T>(buffer, 0, tempLengths, batch.Strides);
            TensorSpan<T> nextHolding = new TensorSpan<T>(buffer, flattenedLength, tempLengths, batch.Strides);
            for (int i = 0; i < sourceToDestinationIndices.Length; i++)
            {
                if (permutation[i] == i)
                    continue;

                int current = i;
                int destination = permutation[current];
                batchView[current].CopyTo(currentHolding);

                while (destination != i)
                {
                    batchView[destination].CopyTo(nextHolding);
                    currentHolding.CopyTo(batchView[destination]);
                    permutation[current] = current;

                    TensorSpan<T> swapTemp = currentHolding;
                    currentHolding = nextHolding;
                    nextHolding = swapTemp;
                    current = destination;
                    destination = permutation[current];
                }

                currentHolding.CopyTo(batchView[i]);
                permutation[current] = current;
            }
        }
        finally
        {
            ArrayPool<int>.Shared.Return(permutation);
            ArrayPool<T>.Shared.Return(buffer);
        }
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
