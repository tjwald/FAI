namespace FAI.Core.Pipelines;

public sealed record BatchRoute<TInput, TOutput>(
    IPipeline<TInput, TOutput> Target,
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
            routes[i].InputIndices.CopyTo(sourceToDestination, currentOffset);
            currentOffset += routes[i].InputIndices.Length;
        }

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

                if (route.Target is IDestinationPipeline<TInput, TOutput> destinationTarget)
                {
                    await destinationTarget.ExecuteAsync(routeInput.Value, destinationSlice, token);
                }
                else
                {
                    TOutput routeOutput = await route.Target.ExecuteAsync(routeInput.Value, token);
                    try
                    {
                        ScatterIdentity(routeOutput, destinationSlice);
                    }
                    finally
                    {
                        await PipelineOutputDisposer.DisposeAsync(routeOutput);
                    }
                }
            },
            cancellationToken);

        _outputBatch.PermuteInPlace(destination, sourceToDestination);
    }

    private void ScatterIdentity(TOutput source, TOutput destination)
    {
        int outputCount = _outputBatch.Count(source);
        Span<int> identity = stackalloc int[outputCount];
        for (int index = 0; index < outputCount; index++)
        {
            identity[index] = index;
        }

        _outputBatch.Scatter(source, destination, identity);
    }

    private static IEnumerable<Range> GetRouteRanges(int count)
    {
        for (int index = 0; index < count; index++)
        {
            yield return index..(index + 1);
        }
    }
}
