using System.Numerics.Tensors;
using FAI.Core.Pipelines;
using FAI.Onnx.Configuration;
using FAI.Onnx.Factories;

namespace FAI.Onnx.ModelExecutors;

public sealed class ConfiguredOnnxModelPipeline : IPipeline<Tensor<long>[], TensorOutputs<float>>
{
    private readonly IPipeline<Tensor<long>[], TensorOutputs<float>> _inner;

    public ConfiguredOnnxModelPipeline(OnnxModelExecutorOptions options)
    {
        _inner = ModelExecutorFactory.CreateModelPipeline(options.ModelExecutorType, options);
    }

    public ValueTask<TensorOutputs<float>> ExecuteAsync(
        Tensor<long>[] input,
        CancellationToken cancellationToken = default)
        => _inner.ExecuteAsync(input, cancellationToken);
}
