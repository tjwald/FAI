using System.Diagnostics.CodeAnalysis;

namespace FAI.Core.Pipelines;

public interface IPipelineChain<TInput, TOutput> : IPipeline<TInput, TOutput>
{
    bool CanWriteOutput { get; }

    ValueTask ExecuteIntoAsync(
        TInput input,
        TOutput output,
        CancellationToken cancellationToken = default);
}

public static class PipelineChain
{
    public static IPipelineChain<TInput, TOutput> Create<TInput, TOutput>(IPipeline<TInput, TOutput> pipeline)
        => pipeline is IPreallocatingPipeline<TInput, TOutput> preallocatingPipeline
            ? new PreallocatingPipelineChain<TInput, TOutput>(preallocatingPipeline)
            : new PipelineChain<TInput, TOutput>(pipeline);
}

public sealed class PipelineChain<TInput, TOutput> : IPipelineChain<TInput, TOutput>
{
    private readonly IPipeline<TInput, TOutput> _pipeline;

    public PipelineChain(IPipeline<TInput, TOutput> pipeline)
    {
        _pipeline = pipeline;
    }

    public bool CanWriteOutput => _pipeline is IPreallocatingPipeline<TInput, TOutput>;

    public ValueTask<TOutput> ExecuteAsync(TInput input, CancellationToken cancellationToken = default)
        => _pipeline.ExecuteAsync(input, cancellationToken);

    public ValueTask ExecuteIntoAsync(
        TInput input,
        TOutput output,
        CancellationToken cancellationToken = default)
    {
        if (_pipeline is not IPreallocatingPipeline<TInput, TOutput> preallocatingPipeline)
        {
            throw new InvalidOperationException("The final pipeline stage does not support destination execution.");
        }

        return preallocatingPipeline.ExecuteAsync(input, output, cancellationToken);
    }
}

internal sealed class PreallocatingPipelineChain<TInput, TOutput> :
    IPipelineChain<TInput, TOutput>,
    IPreallocatingPipeline<TInput, TOutput>
{
    private readonly IPreallocatingPipeline<TInput, TOutput> _pipeline;

    public PreallocatingPipelineChain(IPreallocatingPipeline<TInput, TOutput> pipeline)
    {
        _pipeline = pipeline;
    }

    public bool CanWriteOutput => true;

    public bool TryAllocateOutput(TInput input, [MaybeNullWhen(false)] out TOutput output)
        => _pipeline.TryAllocateOutput(input, out output);

    public ValueTask<TOutput> ExecuteAsync(TInput input, CancellationToken cancellationToken = default)
        => _pipeline.ExecuteAsync(input, cancellationToken);

    public ValueTask ExecuteAsync(
        TInput input,
        TOutput output,
        CancellationToken cancellationToken = default)
        => _pipeline.ExecuteAsync(input, output, cancellationToken);

    public ValueTask ExecuteIntoAsync(
        TInput input,
        TOutput output,
        CancellationToken cancellationToken = default)
        => _pipeline.ExecuteAsync(input, output, cancellationToken);
}

public sealed class AppendedPipelineChain<TInput, TMiddle, TOutput> : IPipelineChain<TInput, TOutput>
{
    private readonly IPipelineChain<TInput, TMiddle> _previous;
    private readonly IPipeline<TMiddle, TOutput> _pipeline;

    public AppendedPipelineChain(IPipelineChain<TInput, TMiddle> previous, IPipeline<TMiddle, TOutput> pipeline)
    {
        _previous = previous;
        _pipeline = pipeline;
    }

    public bool CanWriteOutput => _pipeline is IPreallocatingPipeline<TMiddle, TOutput>;

    public async ValueTask<TOutput> ExecuteAsync(TInput input, CancellationToken cancellationToken = default)
    {
        TMiddle intermediate = await _previous.ExecuteAsync(input, cancellationToken);
        try
        {
            return await _pipeline.ExecuteAsync(intermediate, cancellationToken);
        }
        finally
        {
            await PipelineOutputDisposer.DisposeAsync(intermediate);
        }
    }

    public async ValueTask ExecuteIntoAsync(
        TInput input,
        TOutput output,
        CancellationToken cancellationToken = default)
    {
        if (_pipeline is not IPreallocatingPipeline<TMiddle, TOutput> preallocatingPipeline)
        {
            throw new InvalidOperationException("The final pipeline stage does not support destination execution.");
        }

        TMiddle intermediate = await _previous.ExecuteAsync(input, cancellationToken);
        try
        {
            await preallocatingPipeline.ExecuteAsync(intermediate, output, cancellationToken);
        }
        finally
        {
            await PipelineOutputDisposer.DisposeAsync(intermediate);
        }
    }
}

internal static class PipelineOutputDisposer
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
