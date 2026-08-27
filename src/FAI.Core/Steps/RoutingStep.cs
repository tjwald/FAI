namespace FAI.Core.Steps;

public sealed record BatchRoute<TInput, TOutput>(
    IAllocatingStep<TInput, TOutput> Target,
    int[] InputIndices);

public interface IBatchRoutingStrategy<TInput, TOutput>
{
    IReadOnlyList<BatchRoute<TInput, TOutput>> Route(TInput input);
}

public sealed class RoutingStep<TInput, TOutput, TInputBatch, TOutputBatch>
    : IAllocatingStep<TInput, TOutput>
    where TInputBatch : IReadOnlyIndexedBatch<TInput, TInputBatch>
    where TOutputBatch : IWritableIndexedBatch<TOutput, TOutputBatch>
{
    private readonly IBatchRoutingStrategy<TInput, TOutput> _routingStrategy;

    public RoutingStep(IBatchRoutingStrategy<TInput, TOutput> routingStrategy)
    {
        _routingStrategy = routingStrategy;
    }

    public async ValueTask<BatchLease<TOutput>> RentOutputAsync(
        TInput input,
        CancellationToken cancellationToken = default)
    {
        IReadOnlyList<BatchRoute<TInput, TOutput>> routes = GetValidatedRoutes(input);
        BatchRoute<TInput, TOutput> firstRoute = routes[0];
        using BatchLease<TInput> routeInput = TInputBatch.Gather(input, firstRoute.InputIndices);
        using BatchLease<TOutput> routeOutput = await firstRoute.Target.RentOutputAsync(routeInput.Value, cancellationToken);
        return TOutputBatch.RentLike(routeOutput.Value, TInputBatch.Count(input));
    }

    public async ValueTask<BatchLease<TOutput>> ExecuteAsync(
        TInput input,
        CancellationToken cancellationToken = default)
    {
        IReadOnlyList<BatchRoute<TInput, TOutput>> routes = GetValidatedRoutes(input);
        BatchRoute<TInput, TOutput> firstRoute = routes[0];
        using BatchLease<TInput> routeInput = TInputBatch.Gather(input, firstRoute.InputIndices);
        using BatchLease<TOutput> routeOutput = await firstRoute.Target.RentOutputAsync(routeInput.Value, cancellationToken);
        BatchLease<TOutput> output = TOutputBatch.RentLike(routeOutput.Value, TInputBatch.Count(input));
        try
        {
            await ExecuteRoutesAsync(input, output.Value, routes, cancellationToken);
            return output;
        }
        catch
        {
            output.Dispose();
            throw;
        }
    }

    public async ValueTask ExecuteAsync(
        TInput input,
        TOutput output,
        CancellationToken cancellationToken = default)
    {
        IReadOnlyList<BatchRoute<TInput, TOutput>> routes = GetValidatedRoutes(input);
        await ExecuteRoutesAsync(input, output, routes, cancellationToken);
    }

    private static async ValueTask ExecuteRoutesAsync(
        TInput input,
        TOutput output,
        IReadOnlyList<BatchRoute<TInput, TOutput>> routes,
        CancellationToken cancellationToken)
    {
        if (TOutputBatch.Count(output) != TInputBatch.Count(input))
        {
            throw new ArgumentException("Input and output batch counts must match.", nameof(output));
        }

        foreach (BatchRoute<TInput, TOutput> route in routes)
        {
            cancellationToken.ThrowIfCancellationRequested();
            using BatchLease<TInput> routeInput = TInputBatch.Gather(input, route.InputIndices);
            using BatchLease<TOutput> routeOutput = TOutputBatch.RentLike(output, route.InputIndices.Length);
            await route.Target.ExecuteAsync(routeInput.Value, routeOutput.Value, cancellationToken);
            TOutputBatch.Scatter(routeOutput.Value, output, route.InputIndices);
        }
    }

    private IReadOnlyList<BatchRoute<TInput, TOutput>> GetValidatedRoutes(TInput input)
    {
        IReadOnlyList<BatchRoute<TInput, TOutput>> routes = _routingStrategy.Route(input);
        int count = TInputBatch.Count(input);
        if (count == 0)
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
