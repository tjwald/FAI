namespace FAI.Core.Pipelines;

internal static class PipelineOutputDisposer
{
    public static async ValueTask DisposeAsync<T>(T value)
    {
        if (value is IAsyncDisposable asyncDisposable) await asyncDisposable.DisposeAsync();
        else if (value is IDisposable disposable) disposable.Dispose();
    }
}
