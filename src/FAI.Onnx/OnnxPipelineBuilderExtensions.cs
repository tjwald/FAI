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
        this PipelineBuilder<TStart, Tensor<long>[]> builder)
        => builder.Then(ResolveOnnxModelPipeline);

    private static IPipeline<Tensor<long>[], TensorOutputs<float>> ResolveOnnxModelPipeline(IServiceProvider serviceProvider)
    {
        IPipeline<Tensor<long>[], TensorOutputs<float>>? existing =
            serviceProvider.GetService<IPipeline<Tensor<long>[], TensorOutputs<float>>>();
        if (existing is not null)
        {
            return existing;
        }

        IModelExecutorOptions executorOptions =
            serviceProvider.GetService<OnnxModelExecutorOptions>()
            ?? serviceProvider.GetService<PooledExecutorOptions<OnnxModelExecutorOptions>>()
            ?? serviceProvider.GetService<MultiDeviceExecutorOptions>()
            ?? serviceProvider.GetRequiredService<IModelExecutorOptions>();

        return ModelExecutorFactory.CreateModelPipeline(executorOptions);
    }
}
