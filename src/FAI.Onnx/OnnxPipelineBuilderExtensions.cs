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
        this PipelineBuilder<TStart, BatchEncode> builder)
        => builder.Then(ResolveOnnxModelPipeline);

    public static PipelineBuilder<TStart, TensorOutputs<float>> ThenOnnxModel<TStart>(
        this PipelineBuilder<TStart, Tensor<long>[]> builder)
        => builder.Then(ResolveLegacyOnnxModelPipeline);

    private static IPipeline<BatchEncode, TensorOutputs<float>> ResolveOnnxModelPipeline(IServiceProvider serviceProvider)
    {
        IPipeline<BatchEncode, TensorOutputs<float>>? existing =
            serviceProvider.GetService<IPipeline<BatchEncode, TensorOutputs<float>>>();
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

    private static IPipeline<Tensor<long>[], TensorOutputs<float>> ResolveLegacyOnnxModelPipeline(IServiceProvider serviceProvider)
    {
        IPipeline<Tensor<long>[], TensorOutputs<float>>? existing =
            serviceProvider.GetService<IPipeline<Tensor<long>[], TensorOutputs<float>>>();
        if (existing is not null)
        {
            return existing;
        }

        return new LegacyTensorArrayOnnxModelPipeline(ResolveOnnxModelPipeline(serviceProvider));
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
