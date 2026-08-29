using System.Diagnostics.CodeAnalysis;

namespace FAI.Core.Pipelines;

public interface IPipeline<in TInput, TOutput>
{
    ValueTask<TOutput> ExecuteAsync(TInput input, CancellationToken cancellationToken = default);
}

public interface IPreallocatingPipeline<in TInput, TOutput> : IPipeline<TInput, TOutput>
{
    bool TryAllocateOutput(TInput input, [MaybeNullWhen(false)] out TOutput output);

    ValueTask ExecuteAsync(
        TInput input,
        TOutput output,
        CancellationToken cancellationToken = default)
        ;
}
