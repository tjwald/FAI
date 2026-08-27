namespace FAI.Core.Steps;

public interface IStepChain<TInput, TOutput> : IStep<TInput, TOutput>
{
    ValueTask ExecuteThenAsync<TNext>(
        TInput input,
        IStep<TOutput, TNext> next,
        TNext output,
        CancellationToken cancellationToken = default);
}

public sealed class StepChain<TInput, TOutput> : IStepChain<TInput, TOutput>
{
    private readonly IAllocatingStep<TInput, TOutput> _step;

    public StepChain(IAllocatingStep<TInput, TOutput> step)
    {
        _step = step;
    }

    public ValueTask ExecuteAsync(TInput input, TOutput output, CancellationToken cancellationToken = default)
        => _step.ExecuteAsync(input, output, cancellationToken);

    public async ValueTask ExecuteThenAsync<TNext>(
        TInput input,
        IStep<TOutput, TNext> next,
        TNext output,
        CancellationToken cancellationToken = default)
    {
        using BatchLease<TOutput> intermediate = await _step.ExecuteAsync(input, cancellationToken);
        await next.ExecuteAsync(intermediate.Value, output, cancellationToken);
    }
}

public sealed class AppendedStepChain<TInput, TMiddle, TOutput> : IStepChain<TInput, TOutput>
{
    private readonly IStepChain<TInput, TMiddle> _previous;
    private readonly IAllocatingStep<TMiddle, TOutput> _step;

    public AppendedStepChain(IStepChain<TInput, TMiddle> previous, IAllocatingStep<TMiddle, TOutput> step)
    {
        _previous = previous;
        _step = step;
    }

    public ValueTask ExecuteAsync(TInput input, TOutput output, CancellationToken cancellationToken = default)
        => _previous.ExecuteThenAsync(input, _step, output, cancellationToken);

    public ValueTask ExecuteThenAsync<TNext>(
        TInput input,
        IStep<TOutput, TNext> next,
        TNext output,
        CancellationToken cancellationToken = default)
    {
        var continuation = new AllocatingContinuationStep<TMiddle, TOutput, TNext>(_step, next);
        return _previous.ExecuteThenAsync(input, continuation, output, cancellationToken);
    }
}

internal sealed class AllocatingContinuationStep<TInput, TMiddle, TOutput> : IStep<TInput, TOutput>
{
    private readonly IAllocatingStep<TInput, TMiddle> _step;
    private readonly IStep<TMiddle, TOutput> _next;

    public AllocatingContinuationStep(IAllocatingStep<TInput, TMiddle> step, IStep<TMiddle, TOutput> next)
    {
        _step = step;
        _next = next;
    }

    public async ValueTask ExecuteAsync(TInput input, TOutput output, CancellationToken cancellationToken = default)
    {
        using BatchLease<TMiddle> intermediate = await _step.ExecuteAsync(input, cancellationToken);
        await _next.ExecuteAsync(intermediate.Value, output, cancellationToken);
    }
}
