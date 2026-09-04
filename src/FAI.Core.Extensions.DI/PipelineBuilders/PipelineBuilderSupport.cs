using FAI.Core.Pipelines;

namespace FAI.Core.Extensions.DI;

internal sealed class IdentityPipeline<T> : IPipeline<T, T>
{
    public ValueTask<T> ExecuteAsync(T input, CancellationToken cancellationToken = default) => ValueTask.FromResult(input);
}
