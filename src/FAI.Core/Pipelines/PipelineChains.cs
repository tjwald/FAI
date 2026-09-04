namespace FAI.Core.Pipelines;

public static class PipelineChain
{
    public static IPipelineChain<TInput, TOutput> Create<TInput, TOutput>(IPipeline<TInput, TOutput> pipeline)
        => pipeline is IPreallocatingPipeline<TInput, TOutput> preallocatingPipeline
            ? new PreallocatingPipelineChain<TInput, TOutput>(preallocatingPipeline)
            : new PipelineChain<TInput, TOutput>(pipeline);
}

public sealed class PipelineChain<TInput, TOutput> : IPipelineChain<TInput, TOutput>, IPreallocatingPipeline<TInput, TOutput>
{
    private readonly IPipeline<TInput, TOutput> _pipeline;

    public PipelineChain(IPipeline<TInput, TOutput> pipeline)
    {
        _pipeline = pipeline;
    }

    public bool CanWriteOutput => _pipeline is IPreallocatingPipeline<TInput, TOutput>;

    public ValueTask<TOutput> ExecuteAsync(TInput input, CancellationToken cancellationToken = default)
        => _pipeline.ExecuteAsync(input, cancellationToken);

    public ValueTask ExecuteAsync(TInput input, TOutput output, CancellationToken cancellationToken = default)
        => ExecuteIntoAsync(input, output, cancellationToken);

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

