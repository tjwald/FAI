namespace FAI.Core.ModelExecutors;

/// <summary>
/// Provides reusable instances of <typeparamref name="T"/>.
/// </summary>
public interface IObjectPool<out T>
{
    T Get();
}
