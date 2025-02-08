using ML.Infra.Configurations.ModelExecutors;
using ML.Infra.Utilities;

namespace ML.Infra.ModelExecutors.Onnx;

public sealed class OnnxModelExecutorObjectPool<T> : IObjectPool<T> where T : IOnnxModelExecutor<T>
{
    private readonly List<T> _onnxModelExecutors;
    private readonly CircularAtomicCounter _current;

    public OnnxModelExecutorObjectPool(string modelDir, PooledExecutorOptions<OnnxModelExecutorOptions> options)
    {
        var factory = new InferenceSessionFactory(modelDir, options.ExecutorConfig);
        _onnxModelExecutors = new List<T>(options.ExecutorCount);
        for (int i = 0; i < options.ExecutorCount; i++)
        {
            _onnxModelExecutors.Add(T.Create(factory.Create(), factory.RunOptions, options.ExecutorConfig));
        }

        _current = new CircularAtomicCounter(_onnxModelExecutors.Count);
    }


    public T Get()
    {
        var onnxModelExecutor = _onnxModelExecutors[_current.Next()];
        return onnxModelExecutor;
    }
}