namespace FAI.Core.Steps;

public sealed record BatchRoute<TInput, TOutput>(
    IStep<TInput, TOutput> Target,
    int[] InputIndices);

public interface IBatchRoutingStrategy<TInput, TOutput>
{
    IReadOnlyList<BatchRoute<TInput, TOutput>> Route(TInput input);
}

public sealed class RoutingStep<TInput, TOutput> : IPreallocatingStep<TInput, TOutput>
{
    private readonly IBatchRoutingStrategy<TInput, TOutput> _routingStrategy;
    private readonly IReadOnlyIndexedBatch<TInput> _inputBatch;
    private readonly IWritableIndexedBatch<TOutput> _outputBatch;

    public RoutingStep(
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
        IReadOnlyList<BatchRoute<TInput, TOutput>> routes = GetValidatedRoutes(input);
        if (routes.All(route => route.Target is IPreallocatingStep<TInput, TOutput>) &&
            ((IPreallocatingStep<TInput, TOutput>)routes[0].Target).TryAllocateOutput(input, out TOutput? allocated))
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
                await StepOutputDisposer.DisposeAsync(output);
                throw;
            }
        }

        IReadOnlyList<BatchRoute<TInput, TOutput>> routes = GetValidatedRoutes(input);
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
                await StepOutputDisposer.DisposeAsync(output);
                throw;
            }
        }
        finally
        {
            foreach (TOutput routeOutput in routeOutputs)
            {
                if (routeOutput is not null)
                {
                    await StepOutputDisposer.DisposeAsync(routeOutput);
                }
            }
        }
    }

    public async ValueTask ExecuteAsync(TInput input, TOutput output, CancellationToken cancellationToken = default)
    {
        IReadOnlyList<BatchRoute<TInput, TOutput>> routes = GetValidatedRoutes(input);
        if (_inputBatch.Count(input) != _outputBatch.Count(output))
        {
            throw new ArgumentException("Input and output batch counts must match.", nameof(output));
        }

        var sourceToDestination = new int[_inputBatch.Count(input)];
        int offset = 0;
        foreach (BatchRoute<TInput, TOutput> route in routes)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var target = (IPreallocatingStep<TInput, TOutput>)route.Target;
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

    private IReadOnlyList<BatchRoute<TInput, TOutput>> GetValidatedRoutes(TInput input)
    {
        IReadOnlyList<BatchRoute<TInput, TOutput>> routes = _routingStrategy.Route(input);
        int count = _inputBatch.Count(input);
        if (count == 0 || routes.Count == 0)
        {
            throw new InvalidOperationException("Routing an empty batch requires an explicit output allocator.");
        }

        var seen = new bool[count];
        foreach (BatchRoute<TInput, TOutput> route in routes)
        {
            foreach (int index in route.InputIndices)
            {
                if ((uint)index >= (uint)count || seen[index])
                {
                    throw new InvalidOperationException("Routes must contain each input index exactly once.");
                }

                seen[index] = true;
            }
        }

        if (seen.Contains(false))
        {
            throw new InvalidOperationException("Routes must contain each input index exactly once.");
        }

        return routes;
    }
}
