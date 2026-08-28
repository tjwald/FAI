using FAI.Core.Steps;
using Microsoft.Extensions.DependencyInjection;

namespace FAI.Core.Extensions.DI.Tests;

public sealed class StepPipelineBuilderTests
{
    [Fact]
    public async Task AddPipeline_ComposesTypedStages()
    {
        var services = new ServiceCollection();

        services
            .AddPipeline<int[]>()
            .Then<long[], ToLongStep>()
            .Then<string[], ToStringStep>()
            .Then<int[], StringLengthStep>()
            .Build("lengths");

        await using ServiceProvider serviceProvider = services.BuildServiceProvider();
        IStep<int[], int[]> pipeline = serviceProvider.GetRequiredKeyedService<IStep<int[], int[]>>("lengths");

        int[] output = await pipeline.ExecuteAsync([3, 42, 100], TestContext.Current.CancellationToken);

        Assert.Equal([1, 2, 3], output);
    }

    [Fact]
    public async Task Then_AppliesDecoratorsToStageInDeclaredOrder()
    {
        var services = new ServiceCollection();
        var calls = new List<string>();

        services
            .AddPipeline<int[]>()
            .Then<long[], ToLongStep>(stage => stage
                .Use((_, inner) => new TrackingStep<int[], long[]>("outer", calls, inner))
                .Use((_, inner) => new TrackingStep<int[], long[]>("inner", calls, inner)))
            .Build();

        await using ServiceProvider serviceProvider = services.BuildServiceProvider();
        IStep<int[], long[]> pipeline = serviceProvider.GetRequiredService<IStep<int[], long[]>>();

        long[] output = await pipeline.ExecuteAsync([7], TestContext.Current.CancellationToken);

        Assert.Equal(["outer", "inner"], calls);
        Assert.Equal([7L], output);
    }

    [Fact]
    public async Task UsePartitioning_PreallocatesFullOutputOnceAndWritesSlices()
    {
        var services = new ServiceCollection();
        services.AddSingleton<IBatchPartitioner<ReadOnlyMemory<int>>>(new FixedMemoryPartitioner(2));

        services
            .AddPipeline<ReadOnlyMemory<int>>()
            .Then<Memory<long>, PartitionedLongStep>(stage => stage
                .UseBatchPartitioning(
                    new ReadOnlyMemoryBatchOperations<int>(),
                    new MemoryBatchOperations<long>()))
            .Build();

        await using ServiceProvider serviceProvider = services.BuildServiceProvider();
        IStep<ReadOnlyMemory<int>, Memory<long>> pipeline =
            serviceProvider.GetRequiredService<IStep<ReadOnlyMemory<int>, Memory<long>>>();

        Memory<long> output = await pipeline.ExecuteAsync(
            new[] { 1, 2, 3, 4, 5 },
            TestContext.Current.CancellationToken);
        PartitionedLongStep inner = serviceProvider.GetRequiredService<PartitionedLongStep>();

        Assert.Equal([1L, 2L, 3L, 4L, 5L], output.ToArray());
        Assert.Equal([1, 2, 2], inner.BatchSizes.Order());
        Assert.Equal(1, inner.AllocationCount);
        Assert.Equal(5, inner.AllocatedLengths.Single());
    }

    [Fact]
    public async Task Then_NestsDecoratedTypedPipelineWithoutEndpointAllocator()
    {
        var services = new ServiceCollection();
        var calls = new List<string>();

        services
            .AddPipeline<int[]>()
            .Then<long[], ToLongStep>()
            .Then(
                nested => nested
                    .Then<string[], ToStringStep>()
                    .Then<int[], StringLengthStep>(),
                stage => stage.Use((_, inner) => new TrackingStep<long[], int[]>("nested", calls, inner)))
            .Build();

        await using ServiceProvider serviceProvider = services.BuildServiceProvider();
        IStep<int[], int[]> pipeline = serviceProvider.GetRequiredService<IStep<int[], int[]>>();

        int[] output = await pipeline.ExecuteAsync([3, 42, 100], TestContext.Current.CancellationToken);

        Assert.Equal([1, 2, 3], output);
        Assert.Equal(["nested"], calls);
    }

    [Fact]
    public async Task Then_NestedPipelineInfersOverloadWithoutCasts()
    {
        var services = new ServiceCollection();

        services
            .AddPipeline<int[]>()
            .Then(nested => nested
                .Then<long[], ToLongStep>()
                .Then<string[], ToStringStep>())
            .Build();

        await using ServiceProvider serviceProvider = services.BuildServiceProvider();
        IStep<int[], string[]> pipeline = serviceProvider.GetRequiredService<IStep<int[], string[]>>();

        string[] output = await pipeline.ExecuteAsync([7, 42], TestContext.Current.CancellationToken);

        Assert.Equal(["7", "42"], output);
    }

    [Fact]
    public async Task Then_EndpointAllocatorWritesThroughTypeChangingNestedPipeline()
    {
        var services = new ServiceCollection();

        services
            .AddPipeline<int[]>()
            .Then(
                nested => nested
                    .Then<long[], ToLongStep>()
                    .Then<string[], PreallocatingToStringStep>(),
                (input, out output) =>
                {
                    output = new string[input.Length];
                    return true;
                })
            .Build();

        await using ServiceProvider serviceProvider = services.BuildServiceProvider();
        IStep<int[], string[]> pipeline = serviceProvider.GetRequiredService<IStep<int[], string[]>>();
        var preallocatingPipeline = Assert.IsAssignableFrom<IPreallocatingStep<int[], string[]>>(pipeline);

        Assert.True(preallocatingPipeline.TryAllocateOutput([7, 42], out string[]? output));
        await preallocatingPipeline.ExecuteAsync([7, 42], output, TestContext.Current.CancellationToken);

        Assert.Equal(["7", "42"], output);
    }

    [Fact]
    public async Task Pipeline_DisposesIntermediateAfterDownstreamCompletes()
    {
        var services = new ServiceCollection();

        services
            .AddPipeline<int>()
            .Then<DisposableValue, DisposableValueStep>()
            .Then<int, ReadDisposableValueStep>()
            .Build();

        await using ServiceProvider serviceProvider = services.BuildServiceProvider();
        IStep<int, int> pipeline = serviceProvider.GetRequiredService<IStep<int, int>>();
        DisposableValueStep first = serviceProvider.GetRequiredService<DisposableValueStep>();

        int output = await pipeline.ExecuteAsync(42, TestContext.Current.CancellationToken);

        Assert.Equal(42, output);
        Assert.True(first.Output!.IsDisposed);
    }

    private sealed class ToLongStep : IStep<int[], long[]>
    {
        public ValueTask<long[]> ExecuteAsync(int[] input, CancellationToken cancellationToken = default)
            => ValueTask.FromResult(input.Select(value => (long)value).ToArray());
    }

    private sealed class ToStringStep : IStep<long[], string[]>
    {
        public ValueTask<string[]> ExecuteAsync(long[] input, CancellationToken cancellationToken = default)
            => ValueTask.FromResult(input.Select(value => value.ToString()).ToArray());
    }

    private sealed class PreallocatingToStringStep : IPreallocatingStep<long[], string[]>
    {
        public bool TryAllocateOutput(long[] input, out string[] output)
        {
            output = new string[input.Length];
            return true;
        }

        public async ValueTask<string[]> ExecuteAsync(long[] input, CancellationToken cancellationToken = default)
        {
            _ = TryAllocateOutput(input, out string[] output);
            await ExecuteAsync(input, output, cancellationToken);
            return output;
        }

        public ValueTask ExecuteAsync(
            long[] input,
            string[] output,
            CancellationToken cancellationToken = default)
        {
            for (int i = 0; i < input.Length; i++)
            {
                output[i] = input[i].ToString();
            }

            return ValueTask.CompletedTask;
        }
    }

    private sealed class StringLengthStep : IStep<string[], int[]>
    {
        public ValueTask<int[]> ExecuteAsync(string[] input, CancellationToken cancellationToken = default)
            => ValueTask.FromResult(input.Select(value => value.Length).ToArray());
    }

    private sealed class TrackingStep<TInput, TOutput>(
        string name,
        List<string> calls,
        IStep<TInput, TOutput> inner) : IStep<TInput, TOutput>
    {
        public ValueTask<TOutput> ExecuteAsync(TInput input, CancellationToken cancellationToken = default)
        {
            calls.Add(name);
            return inner.ExecuteAsync(input, cancellationToken);
        }
    }

    private sealed class PartitionedLongStep : IPreallocatingStep<ReadOnlyMemory<int>, Memory<long>>
    {
        private readonly Lock _lock = new();

        public List<int> BatchSizes { get; } = [];

        public List<int> AllocatedLengths { get; } = [];

        public int AllocationCount { get; private set; }

        public bool TryAllocateOutput(ReadOnlyMemory<int> input, out Memory<long> output)
        {
            lock (_lock)
            {
                AllocationCount++;
                AllocatedLengths.Add(input.Length);
            }

            output = new long[input.Length];
            return true;
        }

        public async ValueTask<Memory<long>> ExecuteAsync(
            ReadOnlyMemory<int> input,
            CancellationToken cancellationToken = default)
        {
            _ = TryAllocateOutput(input, out Memory<long> output);
            await ExecuteAsync(input, output, cancellationToken);
            return output;
        }

        public ValueTask ExecuteAsync(
            ReadOnlyMemory<int> input,
            Memory<long> output,
            CancellationToken cancellationToken = default)
        {
            lock (_lock)
            {
                BatchSizes.Add(input.Length);
            }

            for (int i = 0; i < input.Length; i++)
            {
                output.Span[i] = input.Span[i];
            }

            return ValueTask.CompletedTask;
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

    private sealed class DisposableValue(int value) : IDisposable
    {
        public int Value { get; } = value;

        public bool IsDisposed { get; private set; }

        public void Dispose() => IsDisposed = true;
    }

    private sealed class DisposableValueStep : IStep<int, DisposableValue>
    {
        public DisposableValue? Output { get; private set; }

        public ValueTask<DisposableValue> ExecuteAsync(int input, CancellationToken cancellationToken = default)
        {
            Output = new DisposableValue(input);
            return ValueTask.FromResult(Output);
        }
    }

    private sealed class ReadDisposableValueStep : IStep<DisposableValue, int>
    {
        public async ValueTask<int> ExecuteAsync(
            DisposableValue input,
            CancellationToken cancellationToken = default)
        {
            await Task.Yield();
            Assert.False(input.IsDisposed);
            return input.Value;
        }
    }
}
