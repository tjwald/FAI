using FAI.Core.Pipelines;

namespace FAI.Core.Extensions.DI;

internal sealed class IdentityPipeline<T> : IPipeline<T, T>
{
    public ValueTask<T> ExecuteAsync(T input, CancellationToken cancellationToken = default) => ValueTask.FromResult(input);
}

internal sealed class PreallocatingPipeline<TInput, TOutput> : IPreallocatingPipeline<TInput, TOutput>
{
    private readonly IPipelineChain<TInput, TOutput> _pipeline;
    private readonly TryAllocatePipelineOutput<TInput, TOutput> _tryAllocateOutput;

    public PreallocatingPipeline(IPipelineChain<TInput, TOutput> pipeline, TryAllocatePipelineOutput<TInput, TOutput> tryAllocateOutput)
    {
        if (!pipeline.CanWriteOutput) throw new InvalidOperationException("A preallocating nested pipeline requires a destination-writing final stage.");
        (_pipeline, _tryAllocateOutput) = (pipeline, tryAllocateOutput);
    }

    public bool TryAllocateOutput(TInput input, out TOutput output) => _tryAllocateOutput(input, out output);
    public ValueTask<TOutput> ExecuteAsync(TInput input, CancellationToken cancellationToken = default) => _pipeline.ExecuteAsync(input, cancellationToken);
    public ValueTask ExecuteAsync(TInput input, TOutput output, CancellationToken cancellationToken = default) => _pipeline.ExecuteIntoAsync(input, output, cancellationToken);
}
