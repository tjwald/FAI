using FAI.Core.Pipelines;
using Microsoft.Extensions.DependencyInjection;

namespace FAI.Core.Extensions.DI.Tests;

public sealed class PipelinePipelineBuilderTests
{
    [Fact]
    public async Task AddPipeline_ComposesTypedStages()
    {
        var services = new ServiceCollection();

        services
            .AddPipeline<int[]>()
            .Then<long[], ToLongPipeline>()
            .Then<string[], ToStringPipeline>()
            .Then<int[], StringLengthPipeline>()
            .Build("lengths");

        await using ServiceProvider serviceProvider = services.BuildServiceProvider();
        IPipeline<int[], int[]> pipeline = serviceProvider.GetRequiredKeyedService<IPipeline<int[], int[]>>("lengths");

        int[] output = await pipeline.ExecuteAsync([3, 42, 100], TestContext.Current.CancellationToken);

        Assert.Equal([1, 2, 3], output);
    }

    [Fact]
    public async Task Use_AppliesDecoratorsToRemainderInDeclaredOrder()
    {
        var services = new ServiceCollection();
        var calls = new List<string>();

        services
            .AddPipeline<int[]>()
            .Then<long[], ToLongPipeline>()
            .Use(new TrackingForwardDecorator<long[]>("outer", calls))
            .Use(new TrackingForwardDecorator<long[]>("inner", calls))
            .Build();

        await using ServiceProvider serviceProvider = services.BuildServiceProvider();
        IPipeline<int[], long[]> pipeline = serviceProvider.GetRequiredService<IPipeline<int[], long[]>>();

        long[] output = await pipeline.ExecuteAsync([7], TestContext.Current.CancellationToken);

        Assert.Equal(["outer", "inner"], calls);
        Assert.Equal([7L], output);
    }

    [Fact]
    public async Task Use_DeclaredBeforeThen_WrapsCompleteTypeChangingRemainder()
    {
        var services = new ServiceCollection();
        var calls = new List<string>();

        services
            .AddPipeline<int[]>()
            .Then<long[], ToLongPipeline>()
            .Use(new TrackingForwardDecorator<long[]>("remainder", calls))
            .Then<string[], ToStringPipeline>()
            .Then<int[], StringLengthPipeline>()
            .Build();

        await using ServiceProvider serviceProvider = services.BuildServiceProvider();
        IPipeline<int[], int[]> pipeline = serviceProvider.GetRequiredService<IPipeline<int[], int[]>>();

        int[] output = await pipeline.ExecuteAsync([7, 42], TestContext.Current.CancellationToken);

        Assert.Equal(["remainder"], calls);
        Assert.Equal([1, 2], output);
    }

    [Fact]
    public async Task Use_AllowsNestedPipelineInRemainder()
    {
        var services = new ServiceCollection();
        var calls = new List<string>();

        services
            .AddPipeline<int[]>()
            .Then<long[], ToLongPipeline>()
            .Use(new TrackingForwardDecorator<long[]>("remainder", calls))
            .Then(nested => nested
                .Then<string[], ToStringPipeline>()
                .Then<int[], StringLengthPipeline>())
            .Build();

        await using ServiceProvider serviceProvider = services.BuildServiceProvider();
        IPipeline<int[], int[]> pipeline = serviceProvider.GetRequiredService<IPipeline<int[], int[]>>();

        int[] output = await pipeline.ExecuteAsync([7, 42], TestContext.Current.CancellationToken);

        Assert.Equal(["remainder"], calls);
        Assert.Equal([1, 2], output);
    }

    [Fact]
    public async Task UsePartitioning_WritesSlicesIntoCallerOutput()
    {
        var services = new ServiceCollection();
        services.AddSingleton<IBatchPartitioner<ReadOnlyMemory<int>>>(new FixedMemoryPartitioner(2));

        services
            .AddPipeline<ReadOnlyMemory<int>>()
            .Then<ReadOnlyMemory<int>, MemoryIdentityPipeline>()
            .Use(new PartitioningForwardDecorator<int>())
            .Then<Memory<long>, PartitionedLongPipeline>()
            .Build();

        await using ServiceProvider serviceProvider = services.BuildServiceProvider();
        IPipeline<ReadOnlyMemory<int>, Memory<long>> pipeline =
            serviceProvider.GetRequiredService<IPipeline<ReadOnlyMemory<int>, Memory<long>>>();
        var destinationPipeline = Assert.IsAssignableFrom<IDestinationPipeline<ReadOnlyMemory<int>, Memory<long>>>(pipeline);

        Memory<long> output = new long[5];
        await destinationPipeline.ExecuteAsync(
            new[] { 1, 2, 3, 4, 5 },
            output,
            TestContext.Current.CancellationToken);
        PartitionedLongPipeline inner = serviceProvider.GetRequiredService<PartitionedLongPipeline>();

        Assert.Equal([1L, 2L, 3L, 4L, 5L], output.ToArray());
        Assert.Equal([1, 2, 2], inner.BatchSizes.Order());
    }

    [Fact]
    public async Task Then_NestsDecoratedTypedPipelineWithoutEndpointAllocator()
    {
        var services = new ServiceCollection();
        var calls = new List<string>();

        services
            .AddPipeline<int[]>()
            .Then<long[], ToLongPipeline>()
            .Then(
                nested => nested
                    .Use(new TrackingForwardDecorator<long[]>("nested", calls))
                    .Then<string[], ToStringPipeline>()
                    .Then<int[], StringLengthPipeline>())
            .Build();

        await using ServiceProvider serviceProvider = services.BuildServiceProvider();
        IPipeline<int[], int[]> pipeline = serviceProvider.GetRequiredService<IPipeline<int[], int[]>>();

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
                .Then<long[], ToLongPipeline>()
                .Then<string[], ToStringPipeline>())
            .Build();

        await using ServiceProvider serviceProvider = services.BuildServiceProvider();
        IPipeline<int[], string[]> pipeline = serviceProvider.GetRequiredService<IPipeline<int[], string[]>>();

        string[] output = await pipeline.ExecuteAsync([7, 42], TestContext.Current.CancellationToken);

        Assert.Equal(["7", "42"], output);
    }

    [Fact]
    public async Task Then_DestinationExecutionWritesThroughTypeChangingNestedPipeline()
    {
        var services = new ServiceCollection();

        services
            .AddPipeline<int[]>()
            .Then(nested => nested
                    .Then<long[], ToLongPipeline>()
                    .Then<string[], PreallocatingToStringPipeline>())
            .Build();

        await using ServiceProvider serviceProvider = services.BuildServiceProvider();
        IPipeline<int[], string[]> pipeline = serviceProvider.GetRequiredService<IPipeline<int[], string[]>>();
        var destinationPipeline = Assert.IsAssignableFrom<IDestinationPipeline<int[], string[]>>(pipeline);

        string[] output = new string[2];
        await destinationPipeline.ExecuteAsync([7, 42], output, TestContext.Current.CancellationToken);

        Assert.Equal(["7", "42"], output);
    }

    [Fact]
    public async Task Pipeline_DisposesIntermediateAfterDownstreamCompletes()
    {
        var services = new ServiceCollection();

        services
            .AddPipeline<int>()
            .Then<DisposableValue, DisposableValuePipeline>()
            .Then<int, ReadDisposableValuePipeline>()
            .Build();

        await using ServiceProvider serviceProvider = services.BuildServiceProvider();
        IPipeline<int, int> pipeline = serviceProvider.GetRequiredService<IPipeline<int, int>>();
        DisposableValuePipeline first = serviceProvider.GetRequiredService<DisposableValuePipeline>();

        int output = await pipeline.ExecuteAsync(42, TestContext.Current.CancellationToken);

        Assert.Equal(42, output);
        Assert.True(first.Output!.IsDisposed);
    }

    private sealed class ToLongPipeline : IPipeline<int[], long[]>
    {
        public ValueTask<long[]> ExecuteAsync(int[] input, CancellationToken cancellationToken = default)
            => ValueTask.FromResult(input.Select(value => (long)value).ToArray());
    }

    private sealed class ToStringPipeline : IPipeline<long[], string[]>
    {
        public ValueTask<string[]> ExecuteAsync(long[] input, CancellationToken cancellationToken = default)
            => ValueTask.FromResult(input.Select(value => value.ToString()).ToArray());
    }

    private sealed class PreallocatingToStringPipeline : IDestinationPipeline<long[], string[]>
    {
        public async ValueTask<string[]> ExecuteAsync(long[] input, CancellationToken cancellationToken = default)
        {
            string[] output = new string[input.Length];
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

    private sealed class StringLengthPipeline : IPipeline<string[], int[]>
    {
        public ValueTask<int[]> ExecuteAsync(string[] input, CancellationToken cancellationToken = default)
            => ValueTask.FromResult(input.Select(value => value.Length).ToArray());
    }

    private sealed class TrackingPipeline<TInput, TOutput>(
        string name,
        List<string> calls,
        IPipeline<TInput, TOutput> inner) : IPipeline<TInput, TOutput>
    {
        public ValueTask<TOutput> ExecuteAsync(TInput input, CancellationToken cancellationToken = default)
        {
            calls.Add(name);
            return inner.ExecuteAsync(input, cancellationToken);
        }
    }

    private sealed class TrackingForwardDecorator<TInput>(string name, List<string> calls) : IForwardPipelineDecorator<TInput>
    {
        public IPipeline<TInput, TOutput> Apply<TOutput>(
            IServiceProvider serviceProvider,
            IPipeline<TInput, TOutput> pipeline)
            => new TrackingPipeline<TInput, TOutput>(name, calls, pipeline);
    }

    private sealed class PartitioningForwardDecorator<T> : IForwardPipelineDecorator<ReadOnlyMemory<T>>
    {
        public IPipeline<ReadOnlyMemory<T>, TOutput> Apply<TOutput>(
            IServiceProvider serviceProvider,
            IPipeline<ReadOnlyMemory<T>, TOutput> pipeline)
            => new PartitioningPipeline<ReadOnlyMemory<T>, TOutput>(
                pipeline,
                serviceProvider.GetRequiredService<IBatchPartitioner<ReadOnlyMemory<T>>>(),
                new ReadOnlyMemoryBatchOperations<T>(),
                IndexedBatchOperations.GetWritable<TOutput>(),
                serviceProvider.GetService<IPartitionScheduler>());
    }

    private sealed class MemoryIdentityPipeline : IPipeline<ReadOnlyMemory<int>, ReadOnlyMemory<int>>
    {
        public ValueTask<ReadOnlyMemory<int>> ExecuteAsync(
            ReadOnlyMemory<int> input,
            CancellationToken cancellationToken = default)
            => ValueTask.FromResult(input);
    }

    private sealed class PartitionedLongPipeline : IDestinationPipeline<ReadOnlyMemory<int>, Memory<long>>
    {
        private readonly Lock _lock = new();

        public List<int> BatchSizes { get; } = [];

        public async ValueTask<Memory<long>> ExecuteAsync(
            ReadOnlyMemory<int> input,
            CancellationToken cancellationToken = default)
        {
            Memory<long> output = new long[input.Length];
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

    private sealed class DisposableValuePipeline : IPipeline<int, DisposableValue>
    {
        public DisposableValue? Output { get; private set; }

        public ValueTask<DisposableValue> ExecuteAsync(int input, CancellationToken cancellationToken = default)
        {
            Output = new DisposableValue(input);
            return ValueTask.FromResult(Output);
        }
    }

    private sealed class ReadDisposableValuePipeline : IPipeline<DisposableValue, int>
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
