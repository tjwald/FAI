using ML.Infra.Abstractions;
using ML.Infra.Configurations.ModelExecutors;
using ML.Infra.ModelExecutors;
using ML.Onnx.Configuration;
using ML.Onnx.ModelExecutors;

namespace ML.Onnx.Factories;

public enum ModelExecutorType
{
    Simple,
    Pooled,
    Async,
    AsyncPooled,
    Tensor,
    TensorPooled,
}

public static class ModelExecutorFactory
{
    public static async ValueTask<IModelExecutor<long, float>> CreateModelExecutor(string modelDir, ModelExecutorType executorType,
        IModelExecutorConfig modelExecutorOptions)
    {
        Console.WriteLine($"Using model executor {executorType}");
        return executorType switch
        {
            ModelExecutorType.Simple => await OnnxModelExecutor.FromPretrained(modelDir, (OnnxModelExecutorOptions)modelExecutorOptions),
            ModelExecutorType.Pooled => new PooledModelExecutor<long, float>(new OnnxModelExecutorObjectPool<OnnxModelExecutor>(modelDir,
                (PooledExecutorOptions<OnnxModelExecutorOptions>)modelExecutorOptions)),
            ModelExecutorType.Async => await AsyncOnnxModelExecutor.FromPretrained(modelDir, (OnnxModelExecutorOptions)modelExecutorOptions),
            ModelExecutorType.AsyncPooled => new PooledModelExecutor<long, float>(new OnnxModelExecutorObjectPool<AsyncOnnxModelExecutor>(modelDir,
                (PooledExecutorOptions<OnnxModelExecutorOptions>)modelExecutorOptions)),
            ModelExecutorType.Tensor => await OnnxModelTensorExecutor.FromPretrained(modelDir, (OnnxModelExecutorOptions)modelExecutorOptions),
            ModelExecutorType.TensorPooled => new PooledModelExecutor<long, float>(new OnnxModelExecutorObjectPool<AsyncOnnxModelExecutor>(modelDir,
                (PooledExecutorOptions<OnnxModelExecutorOptions>)modelExecutorOptions)),
            _ => throw new NotImplementedException(nameof(executorType)),
        };
    }
}