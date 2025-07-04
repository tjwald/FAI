using FAI.Core.Abstractions;
using FAI.Core.Configurations.ModelExecutors;
using FAI.Core.ModelExecutors;
using FAI.Onnx.Configuration;
using FAI.Onnx.ModelExecutorPools;
using FAI.Onnx.ModelExecutors;

namespace FAI.Onnx.Factories;

/// <summary>
/// Factory class for creating instances of model executors based on configuration.
/// </summary>
public static class ModelExecutorFactory
{
    /// <summary>
    /// Creates an instance of <see cref="IModelExecutor{TInput, TOutput}"/> based on the specified executor type and configuration.
    /// </summary>
    /// <param name="executorType">The type of model executor to create.</param>
    /// <param name="modelExecutorOptions">The configuration options for the model executor.</param>
    /// <returns>Task that creates a model executor</returns>
    /// <exception cref="NotImplementedException">
    /// Thrown when the specified <paramref name="executorType"/> is not implemented.
    /// </exception>
    public static async ValueTask<IModelExecutor<long, float>> CreateModelExecutor(
        ModelExecutorType executorType,
        IModelExecutorConfig modelExecutorOptions)
    {
        switch (modelExecutorOptions)
        {
            case MultiDeviceExecutorOptions multiDeviceExecutorOptions:
                List<Task<OnnxModelExecutorBase>> tasks = multiDeviceExecutorOptions.ExecutorOptions
                    .Select(options => CreateOnnxModelExecutor(executorType, options)).ToList();
                await Task.WhenAll(tasks);

                List<OnnxModelExecutorBase> executors = tasks.Select(modelCreationTask => modelCreationTask.Result).ToList();

                return new PooledModelExecutor<long, float>(new MultiDeviceObjectPool(executors));
            case PooledExecutorOptions<OnnxModelExecutorOptions> pooledExecutorOptions:
            {
                Console.WriteLine($"Using pooling for {executorType}");
                IObjectPool<IModelExecutor<long, float>> objectPool = executorType switch
                {
                    ModelExecutorType.Simple => new OnnxModelExecutorObjectPool<OnnxModelExecutor>(pooledExecutorOptions),
                    ModelExecutorType.Async => new OnnxModelExecutorObjectPool<AsyncOnnxModelExecutor>(pooledExecutorOptions),
                    ModelExecutorType.Tensor => new OnnxModelExecutorObjectPool<OnnxModelTensorExecutor>(pooledExecutorOptions),
                    _ => throw new NotImplementedException(nameof(executorType)),
                };
                return new PooledModelExecutor<long, float>(objectPool);
            }
        }

        return await CreateOnnxModelExecutor(executorType, (OnnxModelExecutorOptions)modelExecutorOptions);
    }

    private static async Task<OnnxModelExecutorBase> CreateOnnxModelExecutor(
        ModelExecutorType executorType,
        OnnxModelExecutorOptions onnxModelExecutorOptions)
    {
        Console.WriteLine($"Using model executor {executorType}");
        return executorType switch
        {
            ModelExecutorType.Simple => await OnnxModelExecutor.FromPretrained(onnxModelExecutorOptions),
            ModelExecutorType.Async => await AsyncOnnxModelExecutor.FromPretrained(onnxModelExecutorOptions),
            ModelExecutorType.Tensor => await OnnxModelTensorExecutor.FromPretrained(onnxModelExecutorOptions),
            _ => throw new NotImplementedException(nameof(executorType)),
        };
    }
}