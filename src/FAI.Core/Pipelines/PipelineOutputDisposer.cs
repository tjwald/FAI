namespace FAI.Core.Pipelines;

using System.Runtime.CompilerServices;

internal static class PipelineOutputDisposer
{
    private static readonly ConditionalWeakTable<object, StrongBox<int>> _borrowed = new();
    private static readonly Lock _lock = new();

    public static void PreserveBorrowed(object? value)
    {
        if (value is null)
        {
            return;
        }

        lock (_lock)
        {
            StrongBox<int> box = _borrowed.GetOrCreateValue(value);
            box.Value++;
        }
    }

    private static bool ConsumeBorrowed(object? value)
    {
        if (value is null)
        {
            return false;
        }

        lock (_lock)
        {
            if (_borrowed.TryGetValue(value, out StrongBox<int>? box) && box.Value > 0)
            {
                box.Value--;
                if (box.Value == 0)
                {
                    _borrowed.Remove(value);
                }

                return true;
            }

            return false;
        }
    }

    public static async ValueTask DisposeAsync<T>(T value)
    {
        if (value is null)
        {
            return;
        }

        if (value is ITuple tuple)
        {
            Exception? firstException = null;
            for (int i = 0; i < tuple.Length; i++)
            {
                object? element = tuple[i];
                if (element is not null && ConsumeBorrowed(element))
                {
                    continue;
                }

                try
                {
                    await DisposeAsync(element);
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

            return;
        }

        if (ConsumeBorrowed(value))
        {
            return;
        }

        if (value is IAsyncDisposable asyncDisposable) await asyncDisposable.DisposeAsync();
        else if (value is IDisposable disposable) disposable.Dispose();
    }
}
