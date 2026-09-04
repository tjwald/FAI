namespace FAI.Core.Pipelines;

public sealed class AppendedPipelineChain<TInput, TMiddle, TOutput> : IPipelineChain<TInput, TOutput>, IPreallocatingPipeline<TInput, TOutput>
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
        try { return await _pipeline.ExecuteAsync(intermediate, cancellationToken); }
        finally { await PipelineOutputDisposer.DisposeAsync(intermediate); }
    }

    public async ValueTask ExecuteAsync(TInput input, TOutput output, CancellationToken cancellationToken = default)
        => await ExecuteIntoAsync(input, output, cancellationToken);

    public async ValueTask ExecuteIntoAsync(TInput input, TOutput output, CancellationToken cancellationToken = default)
    {
        if (_pipeline is not IPreallocatingPipeline<TMiddle, TOutput> preallocatingPipeline)
            throw new InvalidOperationException("The final pipeline stage does not support destination execution.");

        TMiddle intermediate = await _previous.ExecuteAsync(input, cancellationToken);
        try { await preallocatingPipeline.ExecuteAsync(intermediate, output, cancellationToken); }
        finally { await PipelineOutputDisposer.DisposeAsync(intermediate); }
    }
}
