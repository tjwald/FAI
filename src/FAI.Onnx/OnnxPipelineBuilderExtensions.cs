using System.Numerics.Tensors;
using FAI.Core.Configurations.ModelExecutors;
using FAI.Core.Extensions.DI;
using FAI.Core.Pipelines;
using FAI.Onnx.Configuration;
using FAI.Onnx.Factories;
using Microsoft.Extensions.DependencyInjection;

namespace FAI.Onnx;

public static class OnnxPipelineBuilderExtensions
{
    public static PipelineBuilder<TStart, TensorOutputs<float>> ThenOnnxModel<TStart>(
        this PipelineBuilder<TStart, NamedTensorCollection> builder)
        => builder.Then(ResolveOnnxModelPipeline);

    public static PipelineBuilder<TStart, TensorOutputs<float>> ThenOnnxModel<TStart>(
        this PipelineBuilder<TStart, Tensor<long>[]> builder)
        => builder.Then(ResolveLegacyOnnxModelPipeline);

    private static IPipeline<NamedTensorCollection, TensorOutputs<float>> ResolveOnnxModelPipeline(IServiceProvider serviceProvider)
    {
        IPipeline<NamedTensorCollection, TensorOutputs<float>>? existing =
            serviceProvider.GetService<IPipeline<NamedTensorCollection, TensorOutputs<float>>>();
        if (existing is not null)
        {
            return existing;
        }

        IModelExecutorOptions executorOptions = ResolveExecutorOptions(serviceProvider);
        return ModelExecutorFactory.CreateModelPipeline(executorOptions);
    }

    private static IPipeline<Tensor<long>[], TensorOutputs<float>> ResolveLegacyOnnxModelPipeline(IServiceProvider serviceProvider)
    {
        IPipeline<Tensor<long>[], TensorOutputs<float>>? existing =
            serviceProvider.GetService<IPipeline<Tensor<long>[], TensorOutputs<float>>>();
        if (existing is not null)
        {
            return existing;
        }

        IModelExecutorOptions executorOptions = ResolveExecutorOptions(serviceProvider);
        return ModelExecutorFactory.CreateLegacyTensorModelPipeline(executorOptions);
    }

    private static IModelExecutorOptions ResolveExecutorOptions(IServiceProvider serviceProvider)
    {
        return serviceProvider.GetService<OnnxModelExecutorOptions>()
               ?? serviceProvider.GetService<PooledExecutorOptions<OnnxModelExecutorOptions>>()
               ?? serviceProvider.GetService<MultiDeviceExecutorOptions>()
               ?? serviceProvider.GetRequiredService<IModelExecutorOptions>();
    }
}
