using System.Numerics.Tensors;
using FAI.Core.Configurations.ModelExecutors;
using FAI.Core.ModelExecutors;
using FAI.Core.Pipelines;
using FAI.Onnx.Configuration;
using FAI.Onnx.ModelExecutorPools;
using FAI.Onnx.ModelExecutors;

namespace FAI.Onnx.Factories;

/// <summary>
/// Factory class for creating instances of model executors based on configuration.
/// </summary>
public static class ModelExecutorFactory
{
    public static IPipeline<Tensor<long>[], TensorOutputs<float>> CreateModelPipeline(
        ModelExecutorType executorType,
        IModelExecutorOptions modelExecutorOptions)
    {
        return modelExecutorOptions switch
        {
            MultiDeviceExecutorOptions multiDeviceOptions => new PooledOnnxModelPipeline(
                new MultiDeviceObjectPool(multiDeviceOptions.ExecutorOptions
                    .Select(options => CreateOnnxModelExecutor(executorType, options))
                    .ToList())),
            PooledExecutorOptions<OnnxModelExecutorOptions> pooledOptions => new PooledOnnxModelPipeline(
                CreateOnnxModelExecutorPool(executorType, pooledOptions)),
            OnnxModelExecutorOptions onnxOptions => CreateOnnxModelExecutor(executorType, onnxOptions),
            _ => throw new NotImplementedException(modelExecutorOptions.GetType().Name),
        };
    }

    private static IObjectPool<OnnxModelExecutorBase> CreateOnnxModelExecutorPool(
        ModelExecutorType executorType,
        PooledExecutorOptions<OnnxModelExecutorOptions> options)
    {
        return executorType switch
        {
            ModelExecutorType.Simple => new OnnxModelExecutorObjectPool<OnnxModelExecutor>(options),
            ModelExecutorType.Async => new OnnxModelExecutorObjectPool<AsyncOnnxModelExecutor>(options),
            ModelExecutorType.Tensor => new OnnxModelExecutorObjectPool<OnnxModelTensorExecutor>(options),
            _ => throw new NotImplementedException(nameof(executorType)),
        };
    }

    private static OnnxModelExecutorBase CreateOnnxModelExecutor(
        ModelExecutorType executorType,
        OnnxModelExecutorOptions onnxModelExecutorOptions)
    {
        Console.WriteLine($"Using model executor {executorType}");
        return executorType switch
        {
            ModelExecutorType.Simple => OnnxModelExecutor.FromPretrained(onnxModelExecutorOptions),
            ModelExecutorType.Async => AsyncOnnxModelExecutor.FromPretrained(onnxModelExecutorOptions),
            ModelExecutorType.Tensor => OnnxModelTensorExecutor.FromPretrained(onnxModelExecutorOptions),
            _ => throw new NotImplementedException(nameof(executorType)),
        };
    }
}
