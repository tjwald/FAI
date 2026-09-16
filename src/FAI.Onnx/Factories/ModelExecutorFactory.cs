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
    public static IPipeline<BatchEncode, TensorOutputs<float>> CreateModelPipeline(
        IModelExecutorOptions modelExecutorOptions)
        => CreateModelPipeline(ResolveModelExecutorType(modelExecutorOptions), modelExecutorOptions);

    public static ModelExecutorType ResolveModelExecutorType(IModelExecutorOptions options)
    {
        return options switch
        {
            OnnxModelExecutorOptions onnx => onnx.ModelExecutorType,
            PooledExecutorOptions<OnnxModelExecutorOptions> pooled => pooled.ExecutorConfig.ModelExecutorType,
            MultiDeviceExecutorOptions multi when multi.ExecutorOptions.Count > 0 => multi.ExecutorOptions[0].ModelExecutorType,
            _ => ModelExecutorType.Simple,
        };
    }

    public static IPipeline<BatchEncode, TensorOutputs<float>> CreateModelPipeline(
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

    [Obsolete("Use CreateModelPipeline(IModelExecutorOptions) returning IPipeline<BatchEncode, TensorOutputs<float>>.")]
    public static IPipeline<Tensor<long>[], TensorOutputs<float>> CreateLegacyTensorModelPipeline(
        IModelExecutorOptions modelExecutorOptions)
    {
        return CreateLegacyTensorModelPipeline(ResolveModelExecutorType(modelExecutorOptions), modelExecutorOptions);
    }

    [Obsolete("Use CreateModelPipeline(ModelExecutorType, IModelExecutorOptions) returning IPipeline<BatchEncode, TensorOutputs<float>>.")]
    public static IPipeline<Tensor<long>[], TensorOutputs<float>> CreateLegacyTensorModelPipeline(
        ModelExecutorType executorType,
        IModelExecutorOptions modelExecutorOptions)
    {
        return new LegacyTensorArrayOnnxModelPipeline(CreateModelPipeline(executorType, modelExecutorOptions));
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

    private sealed class LegacyTensorArrayOnnxModelPipeline(IPipeline<BatchEncode, TensorOutputs<float>> inner)
        : IPipeline<Tensor<long>[], TensorOutputs<float>>
    {
        public ValueTask<TensorOutputs<float>> ExecuteAsync(
            Tensor<long>[] input,
            CancellationToken cancellationToken = default)
        {
            return inner.ExecuteAsync(BatchEncode.FromArray(input), cancellationToken);
        }
    }
}
