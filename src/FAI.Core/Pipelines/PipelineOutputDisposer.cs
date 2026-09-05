namespace FAI.Core.Pipelines;

internal static class PipelineOutputDisposer
{
    public static async ValueTask DisposeAsync<T>(T value)
    {
        if (value is IAsyncDisposable asyncDisposable) await asyncDisposable.DisposeAsync();
        else if (value is IDisposable disposable) disposable.Dispose();
        else if (value is System.Runtime.CompilerServices.ITuple tuple)
        {
            Exception? firstException = null;
            for (int i = 0; i < tuple.Length; i++)
            {
                try
                {
                    await DisposeAsync(tuple[i]);
                }
                catch (Exception ex)
                {
                    firstException ??= ex;
                }
            }

            if (firstException is not null)
            {
                System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(firstException).Throw();
            }
        }
    }
}
