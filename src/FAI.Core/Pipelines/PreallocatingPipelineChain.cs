namespace FAI.Core.Pipelines;

internal sealed class PreallocatingPipelineChain<TInput, TOutput> : IPipelineChain<TInput, TOutput>, IPreallocatingPipeline<TInput, TOutput>
{
    private readonly IPreallocatingPipeline<TInput, TOutput> _pipeline;

    public PreallocatingPipelineChain(IPreallocatingPipeline<TInput, TOutput> pipeline) => _pipeline = pipeline;

    public bool CanWriteOutput => true;

    public ValueTask<TOutput> ExecuteAsync(TInput input, CancellationToken cancellationToken = default) => _pipeline.ExecuteAsync(input, cancellationToken);

    public ValueTask ExecuteAsync(TInput input, TOutput output, CancellationToken cancellationToken = default) => _pipeline.ExecuteAsync(input, output, cancellationToken);

    public ValueTask ExecuteIntoAsync(TInput input, TOutput output, CancellationToken cancellationToken = default) => _pipeline.ExecuteAsync(input, output, cancellationToken);
}
