using ML.Infra.Configurations.ModelExecutors;
using ML.Infra.ModelExecutors;
using ML.Infra.Utilities;
using ML.Onnx.Configuration;

namespace ML.Onnx.ModelExecutors;

public sealed class OnnxModelExecutorObjectPool<T> : IObjectPool<T> where T : IOnnxModelExecutor<T>
{
    private readonly List<T> _onnxModelExecutors;
    private readonly CircularAtomicCounter _current;

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


    public T Get()
    {
        var onnxModelExecutor = _onnxModelExecutors[_current.Next()];
        return onnxModelExecutor;
    }
}