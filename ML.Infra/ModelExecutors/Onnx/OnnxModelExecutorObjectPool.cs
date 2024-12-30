using ML.Infra.Utilities;

namespace ML.Infra.ModelExecutors.Onnx;

public class OnnxModelExecutorObjectPool<T>: IObjectPool<T> where T : IOnnxModelExecutor<T>
{
    private readonly List<T> _onnxModelExecutors;
    private readonly CircularAtomicCounter _current;

    public OnnxModelExecutorObjectPool(string modelDir, OnnxModelExecutorOptions options)
    {
        var factory = new InferenceSessionFactory(modelDir, options);
        _onnxModelExecutors = new List<T>(options.MaxInferenceSessions);
        for (int i = 0; i < options.MaxInferenceSessions; i++)
        {
            _onnxModelExecutors.Add(T.Create(factory.Create(), factory.RunOptions, options));
        }

        _current = new CircularAtomicCounter(_onnxModelExecutors.Count);
    }


    public T Get()
    {
        var onnxModelExecutor = _onnxModelExecutors[_current.Next()];
        return onnxModelExecutor;
    }
}