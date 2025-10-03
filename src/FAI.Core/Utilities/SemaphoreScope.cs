namespace FAI.Core.Utilities;

/// <summary>
/// Represents a scope for a semaphore, ensuring that the semaphore is released when the scope is disposed.
/// </summary>
public sealed class SemaphoreScope : IDisposable
{
    private readonly SemaphoreSlim _semaphore;

    /// <summary>
    /// Initializes a new instance of the <see cref="SemaphoreScope"/> class.
    /// </summary>
    /// <param name="semaphore">The semaphore to manage within the scope.</param>
    public SemaphoreScope(SemaphoreSlim semaphore)
    {
        _semaphore = semaphore;
    }

    /// <summary>
    /// Releases the semaphore when the scope is disposed.
    /// </summary>
    public void Dispose()
    {
        _semaphore.Release();
    }
}

/// <summary>
/// Represents an empty disposable scope that performs no action on disposal.
/// </summary>
file sealed class EmptyScope : IDisposable
{
#pragma warning disable IDE1006
    // ReSharper disable once InconsistentNaming
    private static readonly Lazy<EmptyScope> _instance = new Lazy<EmptyScope>(() => new EmptyScope());
#pragma warning restore IDE1006

    /// <summary>
    /// Gets the singleton instance of the <see cref="EmptyScope"/> class.
    /// </summary>
    public static EmptyScope Instance => _instance.Value;

    /// <summary>
    /// Performs no action on disposal.
    /// </summary>
    public void Dispose()
    {
    }
}

/// <summary>
/// Provides extension methods for working with <see cref="SemaphoreSlim"/> instances.
/// </summary>
public static class SemaphoreScopeExtensions
{
    /// <summary>
    /// Enters a semaphore scope asynchronously, ensuring the semaphore is released when the scope is disposed.
    /// </summary>
    /// <param name="semaphore">The semaphore to enter. If null, an empty scope is returned.</param>
    /// <returns>A disposable scope that releases the semaphore when disposed.</returns>
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
