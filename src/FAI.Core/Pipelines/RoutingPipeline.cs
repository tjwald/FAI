namespace FAI.Core.Pipelines;

public sealed record BatchRoute<TInput, TOutput>(
    IDestinationPipeline<TInput, TOutput> Target,
    int[] InputIndices);

public sealed class RoutingPipeline<TInput, TOutput> : IDestinationPipeline<TInput, TOutput>
{
    private readonly IBatchRoutingStrategy<TInput, TOutput> _routingStrategy;
    private readonly IReadOnlyIndexedBatch<TInput> _inputBatch;
    private readonly IWritableIndexedBatch<TOutput> _outputBatch;
    private readonly IPartitionScheduler _scheduler;

    public RoutingPipeline(
        IBatchRoutingStrategy<TInput, TOutput> routingStrategy,
        IReadOnlyIndexedBatch<TInput> inputBatch,
        IWritableIndexedBatch<TOutput> outputBatch,
        IPartitionScheduler? scheduler = null)
    {
        _routingStrategy = routingStrategy;
        _inputBatch = inputBatch;
        _outputBatch = outputBatch;
        _scheduler = scheduler ?? new SerialPartitionScheduler();
    }

    public async ValueTask<TOutput> ExecuteAsync(TInput input, CancellationToken cancellationToken = default)
    {
        List<BatchRoute<TInput, TOutput>> routes = _routingStrategy.Route(input);
        if (routes.Count == 0)
        {
            throw new InvalidOperationException("Routing requires at least one route.");
        }

        var routeOutputs = new TOutput[routes.Count];
        try
        {
            await _scheduler.ExecuteAsync(
                GetRouteRanges(routes.Count),
                async (range, token) =>
                {
                    BatchRoute<TInput, TOutput> route = routes[range.Start.Value];
                    using BatchLease<TInput> routeInput = _inputBatch.Gather(input, route.InputIndices);
                    routeOutputs[range.Start.Value] = await route.Target.ExecuteAsync(routeInput.Value, token);
                },
                cancellationToken);

            TOutput output = _outputBatch.AllocateLike(routeOutputs[0], _inputBatch.Count(input));
            try
            {
                for (int index = 0; index < routes.Count; index++)
                {
                    _outputBatch.Scatter(routeOutputs[index], output, routes[index].InputIndices);
                }

                return output;
            }
            catch
            {
                await PipelineOutputDisposer.DisposeAsync(output);
                throw;
            }
        }
        finally
        {
            foreach (TOutput routeOutput in routeOutputs)
            {
                if (routeOutput is not null)
                {
                    await PipelineOutputDisposer.DisposeAsync(routeOutput);
                }
            }
        }
    }

    public async ValueTask ExecuteAsync(TInput input, TOutput destination, CancellationToken cancellationToken = default)
    {
        int totalCount = _inputBatch.Count(input);
        if (totalCount != _outputBatch.Count(destination))
        {
            throw new ArgumentException("Input and output batch counts must match.", nameof(destination));
        }

        if (totalCount == 0)
        {
            return;
        }

        List<BatchRoute<TInput, TOutput>> routes = _routingStrategy.Route(input);
        if (routes.Count == 0)
        {
            throw new InvalidOperationException("Routing requires at least one route.");
        }

        var routeOffsets = new int[routes.Count];
        var sourceToDestination = new int[totalCount];
        int currentOffset = 0;
        for (int i = 0; i < routes.Count; i++)
        {
            routeOffsets[i] = currentOffset;
            int[] indices = routes[i].InputIndices;
            if (currentOffset + indices.Length > totalCount)
            {
                throw new InvalidOperationException(
                    $"Routing routes cover more than the total batch count of {totalCount}.");
            }

            indices.CopyTo(sourceToDestination, currentOffset);
            currentOffset += indices.Length;
        }

        if (currentOffset != totalCount)
        {
            throw new InvalidOperationException(
                $"Routing routes must cover all {totalCount} input items, but routes only covered {currentOffset} items.");
        }

        ValidatePermutation(sourceToDestination, totalCount);

        await _scheduler.ExecuteAsync(
            GetRouteRanges(routes.Count),
            async (range, token) =>
            {
                int routeIndex = range.Start.Value;
                BatchRoute<TInput, TOutput> route = routes[routeIndex];
                if (route.InputIndices.Length == 0)
                {
                    return;
                }

                using BatchLease<TInput> routeInput = _inputBatch.Gather(input, route.InputIndices);
                Range destinationRange = routeOffsets[routeIndex]..(routeOffsets[routeIndex] + route.InputIndices.Length);
                TOutput destinationSlice = _outputBatch.Slice(destination, destinationRange);

                await route.Target.ExecuteAsync(routeInput.Value, destinationSlice, token);
            },
            cancellationToken);

        _outputBatch.PermuteInPlace(destination, sourceToDestination);
    }

    private static void ValidatePermutation(ReadOnlySpan<int> indices, int totalCount)
    {
        Span<bool> seen = totalCount <= 256 ? stackalloc bool[totalCount] : new bool[totalCount];
        for (int i = 0; i < indices.Length; i++)
        {
            int index = indices[i];
            if ((uint)index >= (uint)totalCount || seen[index])
            {
                throw new InvalidOperationException(
                    $"Routing routes must cover every input index from 0 to {totalCount - 1} exactly once. Invalid or duplicate index: {index}.");
            }

            seen[index] = true;
        }
    }

    private static IEnumerable<Range> GetRouteRanges(int count)
    {
        for (int index = 0; index < count; index++)
        {
            yield return index..(index + 1);
        }
    }
}
