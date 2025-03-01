namespace ML.Infra.Utilities;

public sealed class SemaphoreScope: IDisposable
{
    private readonly SemaphoreSlim _semaphore;

    public SemaphoreScope(SemaphoreSlim semaphore)
    {
        _semaphore = semaphore;
    }

    public void Dispose()
    {
        _semaphore.Release();
    }
}

file sealed class EmptyScope: IDisposable
{
    private static Lazy<EmptyScope> _instance = new Lazy<EmptyScope>(() => new EmptyScope());
    public static EmptyScope Instance => _instance.Value;

    public void Dispose()
    {
        
    }
}

public static class SemaphoreScopeExtensions
{
    public static async Task<IDisposable> EnterScope(this SemaphoreSlim? semaphore)
    {
        if (semaphore is null)
        {
            return EmptyScope.Instance;
        }
        await semaphore.WaitAsync();
        return new SemaphoreScope(semaphore);
    }
}