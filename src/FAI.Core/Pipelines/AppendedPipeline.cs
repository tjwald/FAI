namespace FAI.Core.Pipelines;

public static class AppendedPipeline
{
    public static IPipeline<TInput, TOutput> Create<TInput, TMiddle, TOutput>(
        IPipeline<TInput, TMiddle> previous,
        IPipeline<TMiddle, TOutput> pipeline)
        => pipeline is IPreallocatingPipeline<TMiddle, TOutput> preallocatingPipeline
            ? new PreallocatingAppendedPipeline<TInput, TMiddle, TOutput>(previous, preallocatingPipeline)
            : new AppendedPipeline<TInput, TMiddle, TOutput>(previous, pipeline);
}

public class AppendedPipeline<TInput, TMiddle, TOutput> : IPipeline<TInput, TOutput>
{
    private readonly IPipeline<TInput, TMiddle> _previous;
    private readonly IPipeline<TMiddle, TOutput> _pipeline;

    public AppendedPipeline(IPipeline<TInput, TMiddle> previous, IPipeline<TMiddle, TOutput> pipeline)
    {
        _previous = previous;
        _pipeline = pipeline;
    }

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
}

public sealed class PreallocatingAppendedPipeline<TInput, TMiddle, TOutput>
    : AppendedPipeline<TInput, TMiddle, TOutput>, IPreallocatingPipeline<TInput, TOutput>
{
    private readonly IPipeline<TInput, TMiddle> _previous;
    private readonly IPreallocatingPipeline<TMiddle, TOutput> _pipeline;

    public PreallocatingAppendedPipeline(
        IPipeline<TInput, TMiddle> previous,
        IPreallocatingPipeline<TMiddle, TOutput> pipeline)
        : base(previous, pipeline)
    {
        _previous = previous;
        _pipeline = pipeline;
    }

    public async ValueTask ExecuteAsync(TInput input, TOutput output, CancellationToken cancellationToken = default)
    {
        TMiddle intermediate = await _previous.ExecuteAsync(input, cancellationToken);
        try
        {
            await _pipeline.ExecuteAsync(intermediate, output, cancellationToken);
        }
        finally
        {
            await PipelineOutputDisposer.DisposeAsync(intermediate);
        }
    }
}
