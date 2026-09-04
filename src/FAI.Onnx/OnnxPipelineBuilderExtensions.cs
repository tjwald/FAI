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
    public static ComposedPipelineBuilder<Tensor<long>[], TensorOutputs<float>> ThenOnnxModel(
        this PipelineBuilder<Tensor<long>[]> builder)
        => builder.Then(ResolveOnnxModelPipeline);

    public static ComposedPipelineBuilder<TStart, TensorOutputs<float>> ThenOnnxModel<TStart>(
        this ComposedPipelineBuilder<TStart, Tensor<long>[]> builder)
        => builder.Then(ResolveOnnxModelPipeline);

    public static DecoratedPipelineBuilder<TStart, TBoundary, TensorOutputs<float>> ThenOnnxModel<TStart, TBoundary>(
        this DecoratedPipelineBuilder<TStart, TBoundary, Tensor<long>[]> builder)
        => builder.Then(ResolveOnnxModelPipeline);

    private static IPipeline<Tensor<long>[], TensorOutputs<float>> ResolveOnnxModelPipeline(IServiceProvider serviceProvider)
    {
        IPipeline<Tensor<long>[], TensorOutputs<float>>? existing =
            serviceProvider.GetService<IPipeline<Tensor<long>[], TensorOutputs<float>>>();
        if (existing is not null)
        {
            return existing;
        }

        OnnxModelExecutorOptions? onnxOptions = serviceProvider.GetService<OnnxModelExecutorOptions>();
        IModelExecutorOptions executorOptions = (IModelExecutorOptions?)onnxOptions
            ?? serviceProvider.GetRequiredService<IModelExecutorOptions>();
        ModelExecutorType executorType = onnxOptions?.ModelExecutorType ?? ModelExecutorType.Simple;

        return ModelExecutorFactory.CreateModelPipeline(executorType, executorOptions);
    }
}
