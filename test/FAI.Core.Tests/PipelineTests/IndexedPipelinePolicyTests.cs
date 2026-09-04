using System.Numerics.Tensors;
using FAI.Core.Configurations;
using FAI.Core.Pipelines;

namespace FAI.Core.Tests.PipelineTests;

public sealed class IndexedPipelinePolicyTests
{
    [Fact]
    public async Task PartitioningPipeline_WritesTensorSlicesIntoCallerOutput()
    {
        var inner = new TensorCopyPipeline();
        var pipeline = new PartitioningPipeline<Tensor<float>, Tensor<float>>(
            inner,
            new FixedTensorPartitioner(2),
            new TensorBatchOperations<float>(),
            new TensorBatchOperations<float>());

        Tensor<float> input = CreateTensor([4, 2], [1, 2, 3, 4, 5, 6, 7, 8]);
        Tensor<float> output = await pipeline.ExecuteAsync(input, TestContext.Current.CancellationToken);

        Assert.Equal([1, 2, 3, 4, 5, 6, 7, 8], output.AsReadOnlyTensorSpan().AsSpan().ToArray());
        Assert.Equal([2, 2], inner.BatchSizes);
    }

    [Fact]
    public async Task PartitioningPipeline_UsesBoundedParallelScheduler()
    {
        var inner = new DelayedMemoryPipeline();
        var pipeline = new PartitioningPipeline<ReadOnlyMemory<int>, Memory<int>>(
                inner,
                new FixedMemoryPartitioner(1),
            new ReadOnlyMemoryBatchOperations<int>(),
            new MemoryBatchOperations<int>(),
                new ParallelPartitionScheduler(new ParallelPartitionSchedulerOptions(MaxConcurrency: 2)));
        int[] input = [1, 2, 3, 4];
        Memory<int> output = await pipeline.ExecuteAsync(input, TestContext.Current.CancellationToken);

        Assert.Equal(input, output.ToArray());
        Assert.Equal(2, inner.MaximumConcurrency);
    }

    [Fact]
    public async Task PartitioningPipeline_FallsBackWhenPreallocationIsUnavailableForInput()
    {
        var inner = new ConditionalMemoryPipeline();
        var pipeline = new PartitioningPipeline<ReadOnlyMemory<int>, Memory<int>>(
            inner,
            new FixedMemoryPartitioner(2),
            new ReadOnlyMemoryBatchOperations<int>(),
            new MemoryBatchOperations<int>());
        int[] input = [1, 2, 3, 4, 5];

        Memory<int> output = await pipeline.ExecuteAsync(input, TestContext.Current.CancellationToken);

        Assert.Equal(input, output.ToArray());
        Assert.Equal(1, inner.PreallocationAttempts);
        Assert.Equal([2, 2, 1], inner.ExecutedBatchSizes);
        Assert.Equal(0, inner.DestinationExecutions);
    }

    [Fact]
    public async Task OrderingPipeline_RestoresOriginalMemoryOrder()
    {
        var inner = new MemoryIdentityPipeline();
        var pipeline = new OrderingPipeline<ReadOnlyMemory<int>, Memory<int>>(
            inner,
            new DescendingOrdering(),
            new ReadOnlyMemoryBatchOperations<int>(),
            new MemoryBatchOperations<int>());

        int[] values = [10, 30, 20];
        Memory<int> output = await pipeline.ExecuteAsync(values, TestContext.Current.CancellationToken);

        Assert.Equal([30, 20, 10], inner.ObservedInput);
        Assert.Equal(values, output.ToArray());
    }

    [Fact]
    public void TensorBatch_SliceIsAViewOfOriginalTensor()
    {
        Tensor<float> tensor = CreateTensor([3, 2], [1, 2, 3, 4, 5, 6]);

        Tensor<float> middleRow = new TensorBatchOperations<float>().Slice(tensor, 1..2);
        middleRow[0, 0] = 99;

        Assert.Equal(99, tensor[1, 0]);
    }

    [Fact]
    public async Task RoutingPipeline_GathersTargetsAndScattersToOriginalOrder()
    {
        var even = new MultiplyPipeline(10);
        var odd = new MultiplyPipeline(100);
        var routing = new ParityRoutingStrategy(even, odd);
        var pipeline = new RoutingPipeline<ReadOnlyMemory<int>, Memory<int>>(
            routing,
            new ReadOnlyMemoryBatchOperations<int>(),
            new MemoryBatchOperations<int>(),
            input => new int[input.Length]);
        int[] input = [1, 2, 3, 4];
        Memory<int> output = await pipeline.ExecuteAsync(input, TestContext.Current.CancellationToken);

        Assert.Equal([100, 20, 300, 40], output.ToArray());
        Assert.Equal(0, even.AllocationCount);
        Assert.Equal(0, odd.AllocationCount);
        Assert.Empty(even.DestinationBatchSizes);
        Assert.Empty(odd.DestinationBatchSizes);
    }

    [Fact]
    public async Task RoutingPipeline_FallsBackToScatteringReturnedOutputs()
    {
        var routing = new ParityRoutingStrategy(new ReturningMultiplyPipeline(10), new ReturningMultiplyPipeline(100));
        var pipeline = new RoutingPipeline<ReadOnlyMemory<int>, Memory<int>>(
            routing,
            new ReadOnlyMemoryBatchOperations<int>(),
            new MemoryBatchOperations<int>(),
            input => new int[input.Length]);
        int[] input = [1, 2, 3, 4];

        Memory<int> output = await pipeline.ExecuteAsync(input, TestContext.Current.CancellationToken);

        Assert.Equal([100, 20, 300, 40], output.ToArray());
    }

    private static Tensor<float> CreateTensor(ReadOnlySpan<nint> lengths, float[] values)
        => Tensor.Create(values, lengths);

    private sealed class TensorCopyPipeline : IPreallocatingPipeline<Tensor<float>, Tensor<float>>
    {
        public List<int> BatchSizes { get; } = [];

        public bool TryAllocateOutput(Tensor<float> input, out Tensor<float> output)
        {
            output = Tensor.CreateFromShape<float>(input.Lengths);
            return true;
        }

        public async ValueTask<Tensor<float>> ExecuteAsync(
            Tensor<float> input,
            CancellationToken cancellationToken = default)
        {
            _ = TryAllocateOutput(input, out Tensor<float> output);
            await ExecuteAsync(input, output, cancellationToken);
            return output;
        }

        public ValueTask ExecuteAsync(Tensor<float> input, Tensor<float> output, CancellationToken cancellationToken = default)
        {
            BatchSizes.Add(new TensorBatchOperations<float>().Count(input));
            input.AsReadOnlyTensorSpan().CopyTo(output.AsTensorSpan());
            return ValueTask.CompletedTask;
        }
    }

    private sealed class FixedTensorPartitioner(int size) : IBatchPartitioner<Tensor<float>>
    {
        public IEnumerable<Range> Partition(Tensor<float> batch)
        {
            int count = new TensorBatchOperations<float>().Count(batch);
            for (int start = 0; start < count; start += size)
            {
                yield return start..Math.Min(start + size, count);
            }
        }
    }

    private sealed class MemoryIdentityPipeline : IPreallocatingPipeline<ReadOnlyMemory<int>, Memory<int>>
    {
        public int[] ObservedInput { get; private set; } = [];

        public bool TryAllocateOutput(ReadOnlyMemory<int> input, out Memory<int> output)
        {
            output = new int[input.Length];
            return true;
        }

        public async ValueTask<Memory<int>> ExecuteAsync(
            ReadOnlyMemory<int> input,
            CancellationToken cancellationToken = default)
        {
            _ = TryAllocateOutput(input, out Memory<int> output);
            await ExecuteAsync(input, output, cancellationToken);
            return output;
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

    private sealed class DelayedMemoryPipeline : IPreallocatingPipeline<ReadOnlyMemory<int>, Memory<int>>
    {
        private int _concurrency;
        private int _maximumConcurrency;

        public int MaximumConcurrency => _maximumConcurrency;

        public bool TryAllocateOutput(ReadOnlyMemory<int> input, out Memory<int> output)
        {
            output = new int[input.Length];
            return true;
        }

        public async ValueTask<Memory<int>> ExecuteAsync(
            ReadOnlyMemory<int> input,
            CancellationToken cancellationToken = default)
        {
            _ = TryAllocateOutput(input, out Memory<int> output);
            await ExecuteAsync(input, output, cancellationToken);
            return output;
        }

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

    private sealed class ConditionalMemoryPipeline : IPreallocatingPipeline<ReadOnlyMemory<int>, Memory<int>>
    {
        public int PreallocationAttempts { get; private set; }

        public int DestinationExecutions { get; private set; }

        public List<int> ExecutedBatchSizes { get; } = [];

        public bool TryAllocateOutput(ReadOnlyMemory<int> input, out Memory<int> output)
        {
            PreallocationAttempts++;
            output = default;
            return false;
        }

        public ValueTask<Memory<int>> ExecuteAsync(
            ReadOnlyMemory<int> input,
            CancellationToken cancellationToken = default)
        {
            ExecutedBatchSizes.Add(input.Length);
            return ValueTask.FromResult<Memory<int>>(input.ToArray());
        }

        public ValueTask ExecuteAsync(
            ReadOnlyMemory<int> input,
            Memory<int> output,
            CancellationToken cancellationToken = default)
        {
            DestinationExecutions++;
            input.CopyTo(output);
            return ValueTask.CompletedTask;
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

    private sealed class MultiplyPipeline(int multiplier) : IPreallocatingPipeline<ReadOnlyMemory<int>, Memory<int>>
    {
        public int AllocationCount { get; private set; }

        public List<int> DestinationBatchSizes { get; } = [];

        public bool TryAllocateOutput(ReadOnlyMemory<int> input, out Memory<int> output)
        {
            AllocationCount++;
            output = new int[input.Length];
            return true;
        }

        public ValueTask<Memory<int>> ExecuteAsync(
            ReadOnlyMemory<int> input,
            CancellationToken cancellationToken = default)
        {
            var output = new int[input.Length];
            Execute(input, output);
            return ValueTask.FromResult<Memory<int>>(output);
        }

        public ValueTask ExecuteAsync(
            ReadOnlyMemory<int> input,
            Memory<int> output,
            CancellationToken cancellationToken = default)
        {
            DestinationBatchSizes.Add(output.Length);
            Execute(input, output);
            return ValueTask.CompletedTask;
        }

        private void Execute(ReadOnlyMemory<int> input, Memory<int> output)
        {
            for (int i = 0; i < input.Length; i++)
            {
                output.Span[i] = input.Span[i] * multiplier;
            }
        }
    }

    private sealed class ReturningMultiplyPipeline(int multiplier) : IPipeline<ReadOnlyMemory<int>, Memory<int>>
    {
        public ValueTask<Memory<int>> ExecuteAsync(
            ReadOnlyMemory<int> input,
            CancellationToken cancellationToken = default)
        {
            var output = new int[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                output[i] = input.Span[i] * multiplier;
            }

            return ValueTask.FromResult<Memory<int>>(output);
        }
    }

    private sealed class ParityRoutingStrategy(
        IPipeline<ReadOnlyMemory<int>, Memory<int>> even,
        IPipeline<ReadOnlyMemory<int>, Memory<int>> odd)
        : IBatchRoutingStrategy<ReadOnlyMemory<int>, Memory<int>>
    {
        public List<BatchRoute<ReadOnlyMemory<int>, Memory<int>>> Route(ReadOnlyMemory<int> input)
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
