using System.Diagnostics.CodeAnalysis;

namespace FAI.Core.Steps;

public interface IStep<in TInput, TOutput>
{
    ValueTask<TOutput> ExecuteAsync(TInput input, CancellationToken cancellationToken = default);
}

public interface IPreallocatingStep<in TInput, TOutput> : IStep<TInput, TOutput>
{
    bool TryAllocateOutput(TInput input, [MaybeNullWhen(false)] out TOutput output);

    ValueTask ExecuteAsync(
        TInput input,
        TOutput output,
        CancellationToken cancellationToken = default)
        ;
}
