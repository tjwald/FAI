using System.Numerics.Tensors;
using FAI.Core;
using FAI.Core.Configurations.ModelExecutors;
using FAI.Core.Pipelines;
using FAI.Onnx.Configuration;
using FAI.Onnx.Factories;
using FAI.Onnx.ModelExecutors;

namespace FAI.Onnx.Tests.Factories;

public class ModelExecutorFactoryTests(OnnxModelFixture fixture) : IClassFixture<OnnxModelFixture>
{
    private readonly string _modelPath = fixture.ModelPath;

    [Fact]
    public async Task CreateModelPipeline_PooledOptions_ExecutesRealInference()
    {
        var onnxOptions = CreateOnnxOptions();
        var options = new PooledExecutorOptions<OnnxModelExecutorOptions>(onnxOptions, 2);
        IPipeline<Tensor<long>[], TensorOutputs<float>> pipeline =
            ModelExecutorFactory.CreateModelPipeline(ModelExecutorType.Async, options);

        await AssertFinitePipelineOutput(pipeline);
    }

    [Fact]
    public async Task CreateModelPipeline_MultiDeviceOptions_ExecutesRealInference()
    {
        var options = new MultiDeviceExecutorOptions()
            .AddOptions(ConfigureModelPath)
            .AddOptions(ConfigureModelPath);
        IPipeline<Tensor<long>[], TensorOutputs<float>> pipeline =
            ModelExecutorFactory.CreateModelPipeline(ModelExecutorType.Simple, options);

        await AssertFinitePipelineOutput(pipeline);
    }

    [Fact]
    public void CreateModelPipeline_ShouldReturnSimplePipeline_WhenOnnxOptionsProvided()
    {
        // Arrange
        var options = new OnnxModelExecutorOptions().ConfigureOnnxOptions(opt =>
        {
            opt.ModelDir = Path.GetDirectoryName(_modelPath)!;
            opt.ModelFileName = Path.GetFileName(_modelPath);
        });

        // Act
        var pipeline = ModelExecutorFactory.CreateModelPipeline(ModelExecutorType.Simple, options);

        // Assert
        Assert.IsType<OnnxModelExecutor>(pipeline);
    }

    [Fact]
    public void CreateModelPipeline_ShouldThrow_WhenUnknownExecutorType()
    {
        // Arrange
        var options = new OnnxModelExecutorOptions();
        var unknownType = (ModelExecutorType)999;

        // Act & Assert
        Assert.Throws<NotImplementedException>(() => ModelExecutorFactory.CreateModelPipeline(unknownType, options));
    }

    private OnnxModelExecutorOptions CreateOnnxOptions()
    {
        var options = new OnnxModelExecutorOptions();
        ConfigureModelPath(options);
        return options;
    }

    private void ConfigureModelPath(OnnxModelExecutorOptions options)
    {
        options.ConfigureOnnxOptions(onnxOptions =>
        {
            onnxOptions.ModelDir = Path.GetDirectoryName(_modelPath)!;
            onnxOptions.ModelFileName = Path.GetFileName(_modelPath);
        });
    }

    private static async Task AssertFinitePipelineOutput(
        IPipeline<Tensor<long>[], TensorOutputs<float>> pipeline)
    {
        Tensor<long>[] input = [Tensor.Create([11L, 22L, 33L], [1, 3])];
        using TensorOutputs<float> output = await pipeline.ExecuteAsync(input, TestContext.Current.CancellationToken);
        Assert.Equal([11.0f, 22.0f, 33.0f], output.GetOutput(0).AsSpan().ToArray());
    }
}
