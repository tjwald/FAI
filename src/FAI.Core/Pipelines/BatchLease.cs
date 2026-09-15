namespace FAI.Core.Pipelines;

public sealed class BatchLease<T> : IDisposable
{
    private Action<T>? _return;

    public BatchLease(T value, Action<T>? @return = null)
    {
        Value = value;
        _return = @return;
    }

    public T Value { get; }

    public void Dispose()
    {
        Action<T>? @return = Interlocked.Exchange(ref _return, null);
        @return?.Invoke(Value);
    }
}
