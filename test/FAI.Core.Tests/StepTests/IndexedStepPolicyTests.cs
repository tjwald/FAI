using System.Numerics.Tensors;
using FAI.Core.Configurations;
using FAI.Core.Steps;

namespace FAI.Core.Tests.StepTests;

public sealed class IndexedStepPolicyTests
{
    [Fact]
    public async Task PartitioningStep_WritesTensorSlicesIntoCallerOutput()
    {
        var inner = new TensorCopyStep();
        var step = new PartitioningStep<
            Tensor<float>,
            Tensor<float>,
            TensorBatchOperations<float>,
            TensorBatchOperations<float>>(inner, new FixedTensorPartitioner(2));

        Tensor<float> input = CreateTensor([4, 2], [1, 2, 3, 4, 5, 6, 7, 8]);
        Tensor<float> output = Tensor.CreateFromShape<float>([4, 2]);

        await step.ExecuteAsync(input, output, TestContext.Current.CancellationToken);

        Assert.Equal([1, 2, 3, 4, 5, 6, 7, 8], output.AsReadOnlyTensorSpan().AsSpan().ToArray());
        Assert.Equal([2, 2], inner.BatchSizes);
    }

    [Fact]
    public async Task PartitioningStep_UsesBoundedParallelScheduler()
    {
        var inner = new DelayedMemoryStep();
        var step = new PartitioningStep<
            ReadOnlyMemory<int>,
            Memory<int>,
            ReadOnlyMemoryBatchOperations<int>,
            MemoryBatchOperations<int>>(
                inner,
                new FixedMemoryPartitioner(1),
                new ParallelPartitionScheduler(new ParallelPartitionSchedulerOptions(MaxConcurrency: 2)));
        int[] input = [1, 2, 3, 4];
        var output = new int[input.Length];

        await step.ExecuteAsync(input, output, TestContext.Current.CancellationToken);

        Assert.Equal(input, output);
        Assert.Equal(2, inner.MaximumConcurrency);
    }

    [Fact]
    public async Task OrderingStep_RestoresOriginalMemoryOrder()
    {
        var inner = new MemoryIdentityStep();
        var step = new OrderingStep<
            ReadOnlyMemory<int>,
            Memory<int>,
            ReadOnlyMemoryBatchOperations<int>,
            MemoryBatchOperations<int>>(inner, new DescendingOrdering());

        int[] values = [10, 30, 20];
        var output = new int[values.Length];

        await step.ExecuteAsync(values, output, TestContext.Current.CancellationToken);

        Assert.Equal([30, 20, 10], inner.ObservedInput);
        Assert.Equal(values, output);
    }

    [Fact]
    public void TensorBatch_SliceIsAViewOfOriginalTensor()
    {
        Tensor<float> tensor = CreateTensor([3, 2], [1, 2, 3, 4, 5, 6]);

        Tensor<float> middleRow = TensorBatchOperations<float>.Slice(tensor, 1..2);
        middleRow[0, 0] = 99;

        Assert.Equal(99, tensor[1, 0]);
    }

    [Fact]
    public async Task RoutingStep_GathersTargetsAndScattersToOriginalOrder()
    {
        var routing = new ParityRoutingStrategy(new MultiplyStep(10), new MultiplyStep(100));
        var step = new RoutingStep<
            ReadOnlyMemory<int>,
            Memory<int>,
            ReadOnlyMemoryBatchOperations<int>,
            MemoryBatchOperations<int>>(routing);
        int[] input = [1, 2, 3, 4];
        var output = new int[input.Length];

        await step.ExecuteAsync(input, output, TestContext.Current.CancellationToken);

        Assert.Equal([100, 20, 300, 40], output);
    }

    private static Tensor<float> CreateTensor(ReadOnlySpan<nint> lengths, float[] values)
        => Tensor.Create(values, lengths);

    private sealed class TensorCopyStep : IAllocatingStep<Tensor<float>, Tensor<float>>
    {
        public List<int> BatchSizes { get; } = [];

        public ValueTask<BatchLease<Tensor<float>>> RentOutputAsync(
            Tensor<float> input,
            CancellationToken cancellationToken = default)
        {
            Tensor<float> output = Tensor.CreateFromShape<float>(input.Lengths);
            return ValueTask.FromResult(new BatchLease<Tensor<float>>(output));
        }

        public ValueTask ExecuteAsync(Tensor<float> input, Tensor<float> output, CancellationToken cancellationToken = default)
        {
            BatchSizes.Add(TensorBatchOperations<float>.Count(input));
            input.AsReadOnlyTensorSpan().CopyTo(output.AsTensorSpan());
            return ValueTask.CompletedTask;
        }
    }

    private sealed class FixedTensorPartitioner(int size) : IBatchPartitioner<Tensor<float>>
    {
        public IEnumerable<Range> Partition(Tensor<float> batch)
        {
            int count = TensorBatchOperations<float>.Count(batch);
            for (int start = 0; start < count; start += size)
            {
                yield return start..Math.Min(start + size, count);
            }
        }
    }

    private sealed class MemoryIdentityStep : IAllocatingStep<ReadOnlyMemory<int>, Memory<int>>
    {
        public int[] ObservedInput { get; private set; } = [];

        public ValueTask<BatchLease<Memory<int>>> RentOutputAsync(
            ReadOnlyMemory<int> input,
            CancellationToken cancellationToken = default)
        {
            var output = new int[input.Length];
            return ValueTask.FromResult(new BatchLease<Memory<int>>(output));
        }

        public ValueTask ExecuteAsync(
            ReadOnlyMemory<int> input,
            Memory<int> output,
            CancellationToken cancellationToken = default)
        {
            ObservedInput = input.ToArray();
            input.CopyTo(output);
            return ValueTask.CompletedTask;
        }
    }

    private sealed class DelayedMemoryStep : IAllocatingStep<ReadOnlyMemory<int>, Memory<int>>
    {
        private int _concurrency;
        private int _maximumConcurrency;

        public int MaximumConcurrency => _maximumConcurrency;

        public ValueTask<BatchLease<Memory<int>>> RentOutputAsync(
            ReadOnlyMemory<int> input,
            CancellationToken cancellationToken = default)
            => ValueTask.FromResult(new BatchLease<Memory<int>>(new int[input.Length]));

        public async ValueTask ExecuteAsync(
            ReadOnlyMemory<int> input,
            Memory<int> output,
            CancellationToken cancellationToken = default)
        {
            int concurrency = Interlocked.Increment(ref _concurrency);
            InterlockedExtensions.Max(ref _maximumConcurrency, concurrency);
            try
            {
                await Task.Delay(50, cancellationToken);
                input.CopyTo(output);
            }
            finally
            {
                Interlocked.Decrement(ref _concurrency);
            }
        }
    }

    private static class InterlockedExtensions
    {
        public static void Max(ref int location, int value)
        {
            int current;
            do
            {
                current = Volatile.Read(ref location);
                if (current >= value)
                {
                    return;
                }
            }
            while (Interlocked.CompareExchange(ref location, value, current) != current);
        }
    }

    private sealed class FixedMemoryPartitioner(int size) : IBatchPartitioner<ReadOnlyMemory<int>>
    {
        public IEnumerable<Range> Partition(ReadOnlyMemory<int> batch)
        {
            for (int start = 0; start < batch.Length; start += size)
            {
                yield return start..Math.Min(start + size, batch.Length);
            }
        }
    }

    private sealed class DescendingOrdering : IIndexOrdering<ReadOnlyMemory<int>>
    {
        public int[] CreateOrder(ReadOnlyMemory<int> batch)
            => Enumerable.Range(0, batch.Length).OrderByDescending(index => batch.Span[index]).ToArray();
    }

    private sealed class MultiplyStep(int multiplier) : IAllocatingStep<ReadOnlyMemory<int>, Memory<int>>
    {
        public ValueTask<BatchLease<Memory<int>>> RentOutputAsync(
            ReadOnlyMemory<int> input,
            CancellationToken cancellationToken = default)
        {
            var output = new int[input.Length];
            return ValueTask.FromResult(new BatchLease<Memory<int>>(output));
        }

        public ValueTask ExecuteAsync(
            ReadOnlyMemory<int> input,
            Memory<int> output,
            CancellationToken cancellationToken = default)
        {
            for (int i = 0; i < input.Length; i++)
            {
                output.Span[i] = input.Span[i] * multiplier;
            }

            return ValueTask.CompletedTask;
        }
    }

    private sealed class ParityRoutingStrategy(
        IAllocatingStep<ReadOnlyMemory<int>, Memory<int>> even,
        IAllocatingStep<ReadOnlyMemory<int>, Memory<int>> odd)
        : IBatchRoutingStrategy<ReadOnlyMemory<int>, Memory<int>>
    {
        public IReadOnlyList<BatchRoute<ReadOnlyMemory<int>, Memory<int>>> Route(ReadOnlyMemory<int> input)
        {
            int[] evenIndices = Enumerable.Range(0, input.Length).Where(index => input.Span[index] % 2 == 0).ToArray();
            int[] oddIndices = Enumerable.Range(0, input.Length).Where(index => input.Span[index] % 2 != 0).ToArray();
            return
            [
                new BatchRoute<ReadOnlyMemory<int>, Memory<int>>(even, evenIndices),
                new BatchRoute<ReadOnlyMemory<int>, Memory<int>>(odd, oddIndices),
            ];
        }
    }
}
