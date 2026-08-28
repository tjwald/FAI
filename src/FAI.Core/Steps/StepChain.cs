using System.Diagnostics.CodeAnalysis;

namespace FAI.Core.Steps;

public interface IStepChain<TInput, TOutput> : IStep<TInput, TOutput>
{
    bool CanWriteOutput { get; }

    ValueTask ExecuteIntoAsync(
        TInput input,
        TOutput output,
        CancellationToken cancellationToken = default);
}

public static class StepChain
{
    public static IStepChain<TInput, TOutput> Create<TInput, TOutput>(IStep<TInput, TOutput> step)
        => step is IPreallocatingStep<TInput, TOutput> preallocatingStep
            ? new PreallocatingStepChain<TInput, TOutput>(preallocatingStep)
            : new StepChain<TInput, TOutput>(step);
}

public sealed class StepChain<TInput, TOutput> : IStepChain<TInput, TOutput>
{
    private readonly IStep<TInput, TOutput> _step;

    public StepChain(IStep<TInput, TOutput> step)
    {
        _step = step;
    }

    public bool CanWriteOutput => _step is IPreallocatingStep<TInput, TOutput>;

    public ValueTask<TOutput> ExecuteAsync(TInput input, CancellationToken cancellationToken = default)
        => _step.ExecuteAsync(input, cancellationToken);

    public ValueTask ExecuteIntoAsync(
        TInput input,
        TOutput output,
        CancellationToken cancellationToken = default)
    {
        if (_step is not IPreallocatingStep<TInput, TOutput> preallocatingStep)
        {
            throw new InvalidOperationException("The final pipeline stage does not support destination execution.");
        }

        return preallocatingStep.ExecuteAsync(input, output, cancellationToken);
    }
}

internal sealed class PreallocatingStepChain<TInput, TOutput> :
    IStepChain<TInput, TOutput>,
    IPreallocatingStep<TInput, TOutput>
{
    private readonly IPreallocatingStep<TInput, TOutput> _step;

    public PreallocatingStepChain(IPreallocatingStep<TInput, TOutput> step)
    {
        _step = step;
    }

    public bool CanWriteOutput => true;

    public bool TryAllocateOutput(TInput input, [MaybeNullWhen(false)] out TOutput output)
        => _step.TryAllocateOutput(input, out output);

    public ValueTask<TOutput> ExecuteAsync(TInput input, CancellationToken cancellationToken = default)
        => _step.ExecuteAsync(input, cancellationToken);

    public ValueTask ExecuteAsync(
        TInput input,
        TOutput output,
        CancellationToken cancellationToken = default)
        => _step.ExecuteAsync(input, output, cancellationToken);

    public ValueTask ExecuteIntoAsync(
        TInput input,
        TOutput output,
        CancellationToken cancellationToken = default)
        => _step.ExecuteAsync(input, output, cancellationToken);
}

public sealed class AppendedStepChain<TInput, TMiddle, TOutput> : IStepChain<TInput, TOutput>
{
    private readonly IStepChain<TInput, TMiddle> _previous;
    private readonly IStep<TMiddle, TOutput> _step;

    public AppendedStepChain(IStepChain<TInput, TMiddle> previous, IStep<TMiddle, TOutput> step)
    {
        _previous = previous;
        _step = step;
    }

    public bool CanWriteOutput => _step is IPreallocatingStep<TMiddle, TOutput>;

    public async ValueTask<TOutput> ExecuteAsync(TInput input, CancellationToken cancellationToken = default)
    {
        TMiddle intermediate = await _previous.ExecuteAsync(input, cancellationToken);
        try
        {
            return await _step.ExecuteAsync(intermediate, cancellationToken);
        }
        finally
        {
            await StepOutputDisposer.DisposeAsync(intermediate);
        }
    }

    public async ValueTask ExecuteIntoAsync(
        TInput input,
        TOutput output,
        CancellationToken cancellationToken = default)
    {
        if (_step is not IPreallocatingStep<TMiddle, TOutput> preallocatingStep)
        {
            throw new InvalidOperationException("The final pipeline stage does not support destination execution.");
        }

        TMiddle intermediate = await _previous.ExecuteAsync(input, cancellationToken);
        try
        {
            await preallocatingStep.ExecuteAsync(intermediate, output, cancellationToken);
        }
        finally
        {
            await StepOutputDisposer.DisposeAsync(intermediate);
        }
    }
}

internal static class StepOutputDisposer
{
    public static async ValueTask DisposeAsync<T>(T value)
    {
        if (value is IAsyncDisposable asyncDisposable)
        {
            await asyncDisposable.DisposeAsync();
        }
        else if (value is IDisposable disposable)
        {
            disposable.Dispose();
        }
    }
}
