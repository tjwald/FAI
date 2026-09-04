namespace FAI.Core.Pipelines;

public sealed record BatchRoute<TInput, TOutput>(
    IPipeline<TInput, TOutput> Target,
    int[] InputIndices);

public sealed class RoutingPipeline<TInput, TOutput> : IPipeline<TInput, TOutput>
{
    private readonly IBatchRoutingStrategy<TInput, TOutput> _routingStrategy;
    private readonly IReadOnlyIndexedBatch<TInput> _inputBatch;
    private readonly IWritableIndexedBatch<TOutput> _outputBatch;
    private readonly Func<TInput, TOutput> _allocateOutput;
    private readonly IPartitionScheduler _scheduler;

    public RoutingPipeline(
        IBatchRoutingStrategy<TInput, TOutput> routingStrategy,
        IReadOnlyIndexedBatch<TInput> inputBatch,
        IWritableIndexedBatch<TOutput> outputBatch,
        Func<TInput, TOutput> allocateOutput,
        IPartitionScheduler? scheduler = null)
    {
        _routingStrategy = routingStrategy;
        _inputBatch = inputBatch;
        _outputBatch = outputBatch;
        _allocateOutput = allocateOutput;
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

            TOutput output = _allocateOutput(input);
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

    private static IEnumerable<Range> GetRouteRanges(int count)
    {
        for (int index = 0; index < count; index++)
        {
            yield return index..(index + 1);
        }
    }
}
