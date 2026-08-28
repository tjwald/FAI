using System.Numerics.Tensors;
using FAI.Core;
using FAI.Core.Configurations.ModelExecutors;
using FAI.Core.Steps;
using FAI.Onnx.Configuration;
using FAI.Onnx.Factories;
using FAI.Onnx.ModelExecutors;

namespace FAI.Onnx.Tests.Factories;

public class ModelExecutorFactoryTests(OnnxModelFixture fixture) : IClassFixture<OnnxModelFixture>
{
    private readonly string _modelPath = fixture.ModelPath;

    [Fact]
    public async Task CreateModelStep_PooledOptions_ExecutesRealInference()
    {
        var onnxOptions = CreateOnnxOptions();
        var options = new PooledExecutorOptions<OnnxModelExecutorOptions>(onnxOptions, 2);
        IStep<Tensor<long>[], TensorOutputs<float>> step =
            ModelExecutorFactory.CreateModelStep(ModelExecutorType.Async, options);

        await AssertFiniteStepOutput(step);
    }

    [Fact]
    public async Task CreateModelStep_MultiDeviceOptions_ExecutesRealInference()
    {
        var options = new MultiDeviceExecutorOptions()
            .AddOptions(ConfigureModelPath)
            .AddOptions(ConfigureModelPath);
        IStep<Tensor<long>[], TensorOutputs<float>> step =
            ModelExecutorFactory.CreateModelStep(ModelExecutorType.Simple, options);

        await AssertFiniteStepOutput(step);
    }

    [Fact]
    public void CreateModelStep_ShouldReturnSimpleStep_WhenOnnxOptionsProvided()
    {
        // Arrange
        var options = new OnnxModelExecutorOptions().ConfigureOnnxOptions(opt =>
        {
            opt.ModelDir = Path.GetDirectoryName(_modelPath)!;
            opt.ModelFileName = Path.GetFileName(_modelPath);
        });

        // Act
        var step = ModelExecutorFactory.CreateModelStep(ModelExecutorType.Simple, options);

        // Assert
        Assert.IsType<OnnxModelExecutor>(step);
    }

    [Fact]
    public void CreateModelStep_ShouldThrow_WhenUnknownExecutorType()
    {
        // Arrange
        var options = new OnnxModelExecutorOptions();
        var unknownType = (ModelExecutorType)999;

        // Act & Assert
        Assert.Throws<NotImplementedException>(() => ModelExecutorFactory.CreateModelStep(unknownType, options));
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

    private static async Task AssertFiniteStepOutput(
        IStep<Tensor<long>[], TensorOutputs<float>> step)
    {
        Tensor<long>[] input = [Tensor.Create([11L, 22L, 33L], [1, 3])];
        using TensorOutputs<float> output = await step.ExecuteAsync(input, TestContext.Current.CancellationToken);
        Assert.Equal([11.0f, 22.0f, 33.0f], output.GetOutput(0).AsSpan().ToArray());
    }
}
