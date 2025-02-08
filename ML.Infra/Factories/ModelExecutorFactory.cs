using ML.Infra.Abstractions;
using ML.Infra.Configurations.ModelExecutors;
using ML.Infra.ModelExecutors;
using ML.Infra.ModelExecutors.Onnx;

namespace ML.Infra.Factories;

public enum ModelExecutorType
{
    Simple,
    Pooled,
    Async,
    AsyncPooled,
}

public static class ModelExecutorFactory
{
    public static async ValueTask<IModelExecutor<long, float>> CreateModelExecutor(string modelDir, ModelExecutorType executorType,
        IModelExecutorConfig modelExecutorOptions)
    {
        return executorType switch
        {
            ModelExecutorType.Simple => await OnnxModelExecutor.FromPretrained(modelDir, (OnnxModelExecutorOptions)modelExecutorOptions),
            ModelExecutorType.Pooled => new PooledModelExecutor<long, float>(new OnnxModelExecutorObjectPool<OnnxModelExecutor>(modelDir,
                (PooledExecutorOptions<OnnxModelExecutorOptions>)modelExecutorOptions)),
            ModelExecutorType.Async => await AsyncOnnxModelExecutor.FromPretrained(modelDir, (OnnxModelExecutorOptions)modelExecutorOptions),
            ModelExecutorType.AsyncPooled => new PooledModelExecutor<long, float>(new OnnxModelExecutorObjectPool<AsyncOnnxModelExecutor>(modelDir,
                (PooledExecutorOptions<OnnxModelExecutorOptions>)modelExecutorOptions)),
            _ => throw new NotImplementedException(nameof(executorType)),
        };
    }
}