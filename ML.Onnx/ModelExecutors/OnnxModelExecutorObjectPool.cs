using ML.Infra.Configurations.ModelExecutors;
using ML.Infra.ModelExecutors;
using ML.Infra.Utilities;
using ML.Onnx.Configuration;
using ML.Onnx.Utils;

namespace ML.Onnx.ModelExecutors;

/// <summary>
/// An executor that delegates to a pool of <see cref="IOnnxModelExecutor{T}"/>
/// </summary>
/// <typeparam name="T">The type of the model executor, implementing <see cref="IOnnxModelExecutor{T}"/>.</typeparam>
public sealed class OnnxModelExecutorObjectPool<T> : IObjectPool<T> where T : IOnnxModelExecutor<T>
{
    private readonly List<T> _onnxModelExecutors;
    private readonly CircularAtomicCounter _current;

    /// <summary>
    /// Initializes a new instance of the <see cref="OnnxModelExecutorObjectPool{T}"/> class using the specified pooled executor options.
    /// </summary>
    /// <param name="options">
    /// The options for creating and configuring the pool, including the number of executors and ONNX options.
    /// </param>
    public OnnxModelExecutorObjectPool(PooledExecutorOptions<OnnxModelExecutorOptions> options)
    {
        var factory = new InferenceSessionFactory(options.ExecutorConfig.OnnxOptions);
        _onnxModelExecutors = new List<T>(options.ExecutorCount);
        for (int i = 0; i < options.ExecutorCount; i++)
        {
            _onnxModelExecutors.Add(T.Create(factory.Create(), factory.RunOptions, options.ExecutorConfig));
        }

        _current = new CircularAtomicCounter(_onnxModelExecutors.Count);
    }

    /// <summary>
    /// Retrieves an instance of the model executor from the pool in a round-robin fashion.
    /// </summary>
    /// <returns>An instance of <typeparamref name="T"/> from the pool.</returns>
    public T Get()
    {
        var onnxModelExecutor = _onnxModelExecutors[_current.Next()];
        return onnxModelExecutor;
    }
}
