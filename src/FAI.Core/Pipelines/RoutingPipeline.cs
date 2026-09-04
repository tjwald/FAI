namespace FAI.Core.Pipelines;

public sealed record BatchRoute<TInput, TOutput>(
    IPipeline<TInput, TOutput> Target,
    int[] InputIndices);

public interface IBatchRoutingStrategy<TInput, TOutput>
{
    List<BatchRoute<TInput, TOutput>> Route(TInput input);
}

public sealed class RoutingPipeline<TInput, TOutput> : IPreallocatingPipeline<TInput, TOutput>
{
    private readonly IBatchRoutingStrategy<TInput, TOutput> _routingStrategy;
    private readonly IReadOnlyIndexedBatch<TInput> _inputBatch;
    private readonly IWritableIndexedBatch<TOutput> _outputBatch;

    public RoutingPipeline(
        IBatchRoutingStrategy<TInput, TOutput> routingStrategy,
        IReadOnlyIndexedBatch<TInput> inputBatch,
        IWritableIndexedBatch<TOutput> outputBatch)
    {
        _routingStrategy = routingStrategy;
        _inputBatch = inputBatch;
        _outputBatch = outputBatch;
    }

    public bool TryAllocateOutput(TInput input, out TOutput output)
    {
        List<BatchRoute<TInput, TOutput>> routes = _routingStrategy.Route(input);
        if (routes.Count > 0 && routes[0].Target is IPreallocatingPipeline<TInput, TOutput> preallocatingPipeline &&
            preallocatingPipeline.TryAllocateOutput(input, out TOutput? allocated))
        {
            output = allocated;
            return true;
        }

        output = default!;
        return false;
    }

    public async ValueTask<TOutput> ExecuteAsync(TInput input, CancellationToken cancellationToken = default)
    {
        if (TryAllocateOutput(input, out TOutput? output))
        {
            try
            {
                await ExecuteAsync(input, output, cancellationToken);
                return output;
            }
            catch
            {
                await PipelineOutputDisposer.DisposeAsync(output);
                throw;
            }
        }

        List<BatchRoute<TInput, TOutput>> routes = _routingStrategy.Route(input);
        if (routes.Count == 0)
        {
            throw new InvalidOperationException("Routing requires at least one route.");
        }
        var routeOutputs = new TOutput[routes.Count];
        try
        {
            for (int i = 0; i < routes.Count; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                BatchRoute<TInput, TOutput> route = routes[i];
                using BatchLease<TInput> routeInput = _inputBatch.Gather(input, route.InputIndices);
                routeOutputs[i] = await route.Target.ExecuteAsync(routeInput.Value, cancellationToken);
            }

            output = _outputBatch.AllocateLike(routeOutputs[0], _inputBatch.Count(input));
            try
            {
                for (int i = 0; i < routes.Count; i++)
                {
                    _outputBatch.Scatter(routeOutputs[i], output, routes[i].InputIndices);
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

    public async ValueTask ExecuteAsync(TInput input, TOutput output, CancellationToken cancellationToken = default)
    {
        List<BatchRoute<TInput, TOutput>> routes = _routingStrategy.Route(input);
        if (_inputBatch.Count(input) != _outputBatch.Count(output))
        {
            throw new ArgumentException("Input and output batch counts must match.", nameof(output));
        }

        var sourceToDestination = new int[_inputBatch.Count(input)];
        int offset = 0;
        foreach (BatchRoute<TInput, TOutput> route in routes)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var target = (IPreallocatingPipeline<TInput, TOutput>)route.Target;
            using BatchLease<TInput> routeInput = _inputBatch.Gather(input, route.InputIndices);
            await target.ExecuteAsync(
                routeInput.Value,
                _outputBatch.Slice(output, offset..(offset + route.InputIndices.Length)),
                cancellationToken);
            route.InputIndices.CopyTo(sourceToDestination, offset);
            offset += route.InputIndices.Length;
        }

        _outputBatch.PermuteInPlace(output, sourceToDestination);
    }

}
