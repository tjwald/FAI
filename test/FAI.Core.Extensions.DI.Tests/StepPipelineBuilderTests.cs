using FAI.Core.Steps;
using Microsoft.Extensions.DependencyInjection;

namespace FAI.Core.Extensions.DI.Tests;

public sealed class StepPipelineBuilderTests
{
    [Fact]
    public async Task AddPipeline_ComposesTypedStagesAndReturnsIntermediateLeases()
    {
        var services = new ServiceCollection();
        services.AddSingleton<LeaseTracker>();

        services
            .AddPipeline<int[]>()
            .Then<long[], ToLongStep>()
            .Then<string[], ToStringStep>()
            .Then<int[], StringLengthStep>()
            .Build("lengths");

        await using ServiceProvider serviceProvider = services.BuildServiceProvider();
        IStep<int[], int[]> pipeline = serviceProvider.GetRequiredKeyedService<IStep<int[], int[]>>("lengths");
        var output = new int[3];

        await pipeline.ExecuteAsync([3, 42, 100], output, TestContext.Current.CancellationToken);

        Assert.Equal([1, 2, 3], output);
        Assert.Equal(2, serviceProvider.GetRequiredService<LeaseTracker>().Returned);
    }

    [Fact]
    public async Task Then_AppliesDecoratorsToStageInDeclaredOrder()
    {
        var services = new ServiceCollection();
        var calls = new List<string>();
        services.AddSingleton<LeaseTracker>();

        services
            .AddPipeline<int[]>()
            .Then<long[], ToLongStep>(stage => stage
                .Use((_, inner) => new TrackingStep<int[], long[]>("outer", calls, inner))
                .Use((_, inner) => new TrackingStep<int[], long[]>("inner", calls, inner)))
            .Build();

        await using ServiceProvider serviceProvider = services.BuildServiceProvider();
        IStep<int[], long[]> pipeline = serviceProvider.GetRequiredService<IStep<int[], long[]>>();
        var output = new long[1];

        await pipeline.ExecuteAsync([7], output, TestContext.Current.CancellationToken);

        Assert.Equal(["outer", "inner"], calls);
        Assert.Equal([7L], output);
    }

    [Fact]
    public async Task UsePartitioning_WrapsOnlyConfiguredStage()
    {
        var services = new ServiceCollection();
        services.AddSingleton<IBatchPartitioner<ReadOnlyMemory<int>>>(new FixedMemoryPartitioner(2));

        services
            .AddPipeline<ReadOnlyMemory<int>>()
            .Then<Memory<long>, PartitionedLongStep>(stage => stage
                .UseBatchPartitioning<ReadOnlyMemoryBatchOperations<int>, MemoryBatchOperations<long>>())
            .Build();

        await using ServiceProvider serviceProvider = services.BuildServiceProvider();
        IStep<ReadOnlyMemory<int>, Memory<long>> pipeline =
            serviceProvider.GetRequiredService<IStep<ReadOnlyMemory<int>, Memory<long>>>();
        var output = new long[5];

        int[] input = [1, 2, 3, 4, 5];
        await pipeline.ExecuteAsync(input, output, TestContext.Current.CancellationToken);

        Assert.Equal([1L, 2L, 3L, 4L, 5L], output);
        Assert.Equal([2, 2, 1], serviceProvider.GetRequiredService<PartitionedLongStep>().BatchSizes);
    }

    [Fact]
    public async Task Then_NestsDecoratedTypedPipelineInsideExistingPipeline()
    {
        var services = new ServiceCollection();
        var calls = new List<string>();
        services.AddSingleton<LeaseTracker>();

        services
            .AddPipeline<int[]>()
            .Then<long[], ToLongStep>()
            .Then(
                nested => nested
                    .Then<string[], ToStringStep>()
                    .Then<int[], StringLengthStep>(),
                (_, input, _) => ValueTask.FromResult(new BatchLease<int[]>(new int[input.Length])),
                stage => stage.Use((_, inner) => new TrackingStep<long[], int[]>("nested", calls, inner)))
            .Build();

        await using ServiceProvider serviceProvider = services.BuildServiceProvider();
        IStep<int[], int[]> pipeline = serviceProvider.GetRequiredService<IStep<int[], int[]>>();
        var output = new int[3];

        await pipeline.ExecuteAsync([3, 42, 100], output, TestContext.Current.CancellationToken);

        Assert.Equal([1, 2, 3], output);
        Assert.Equal(["nested"], calls);
        Assert.Equal(2, serviceProvider.GetRequiredService<LeaseTracker>().Returned);
    }

    [Fact]
    public async Task Then_NestedPipelineWithoutDecoratorInfersOverloadWithoutCasts()
    {
        var services = new ServiceCollection();
        services.AddSingleton<LeaseTracker>();

        services
            .AddPipeline<int[]>()
            .Then(
                nested => nested
                    .Then<long[], ToLongStep>()
                    .Then<string[], ToStringStep>(),
                (_, input, _) => ValueTask.FromResult(new BatchLease<string[]>(new string[input.Length])))
            .Build();

        await using ServiceProvider serviceProvider = services.BuildServiceProvider();
        IStep<int[], string[]> pipeline = serviceProvider.GetRequiredService<IStep<int[], string[]>>();
        var output = new string[2];

        await pipeline.ExecuteAsync([7, 42], output, TestContext.Current.CancellationToken);

        Assert.Equal(["7", "42"], output);
    }

    private sealed class LeaseTracker
    {
        public int Returned { get; private set; }

        public void Return() => Returned++;
    }

    private sealed class ToLongStep(LeaseTracker tracker) : IAllocatingStep<int[], long[]>
    {
        public ValueTask<BatchLease<long[]>> RentOutputAsync(int[] input, CancellationToken cancellationToken = default)
            => ValueTask.FromResult(new BatchLease<long[]>(new long[input.Length], _ => tracker.Return()));

        public ValueTask ExecuteAsync(int[] input, long[] output, CancellationToken cancellationToken = default)
        {
            for (int i = 0; i < input.Length; i++)
            {
                output[i] = input[i];
            }

            return ValueTask.CompletedTask;
        }
    }

    private sealed class ToStringStep(LeaseTracker tracker) : IAllocatingStep<long[], string[]>
    {
        public ValueTask<BatchLease<string[]>> RentOutputAsync(long[] input, CancellationToken cancellationToken = default)
            => ValueTask.FromResult(new BatchLease<string[]>(new string[input.Length], _ => tracker.Return()));

        public ValueTask ExecuteAsync(long[] input, string[] output, CancellationToken cancellationToken = default)
        {
            for (int i = 0; i < input.Length; i++)
            {
                output[i] = input[i].ToString();
            }

            return ValueTask.CompletedTask;
        }
    }

    private sealed class StringLengthStep : IAllocatingStep<string[], int[]>
    {
        public ValueTask<BatchLease<int[]>> RentOutputAsync(string[] input, CancellationToken cancellationToken = default)
            => ValueTask.FromResult(new BatchLease<int[]>(new int[input.Length]));

        public ValueTask ExecuteAsync(string[] input, int[] output, CancellationToken cancellationToken = default)
        {
            for (int i = 0; i < input.Length; i++)
            {
                output[i] = input[i].Length;
            }

            return ValueTask.CompletedTask;
        }
    }

    private sealed class TrackingStep<TInput, TOutput>(
        string name,
        List<string> calls,
        IAllocatingStep<TInput, TOutput> inner) : IAllocatingStep<TInput, TOutput>
    {
        public ValueTask<BatchLease<TOutput>> RentOutputAsync(TInput input, CancellationToken cancellationToken = default)
            => inner.RentOutputAsync(input, cancellationToken);

        public ValueTask ExecuteAsync(TInput input, TOutput output, CancellationToken cancellationToken = default)
        {
            calls.Add(name);
            return inner.ExecuteAsync(input, output, cancellationToken);
        }
    }

    private sealed class PartitionedLongStep : IAllocatingStep<ReadOnlyMemory<int>, Memory<long>>
    {
        public List<int> BatchSizes { get; } = [];

        public ValueTask<BatchLease<Memory<long>>> RentOutputAsync(
            ReadOnlyMemory<int> input,
            CancellationToken cancellationToken = default)
        {
            var output = new long[input.Length];
            return ValueTask.FromResult(new BatchLease<Memory<long>>(output));
        }

        public ValueTask ExecuteAsync(
            ReadOnlyMemory<int> input,
            Memory<long> output,
            CancellationToken cancellationToken = default)
        {
            BatchSizes.Add(input.Length);
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
}
