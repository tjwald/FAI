using System.Numerics.Tensors;
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
    public async Task CreateMaterializingModelStep_PooledOptions_ExecutesRealInference()
    {
        var onnxOptions = CreateOnnxOptions();
        var options = new PooledExecutorOptions<OnnxModelExecutorOptions>(onnxOptions, 2);
        IAllocatingStep<Tensor<long>[], Tensor<float>[]> step =
            ModelExecutorFactory.CreateMaterializingModelStep(ModelExecutorType.Async, options);

        await AssertFiniteStepOutput(step);
    }

    [Fact]
    public async Task CreateMaterializingModelStep_MultiDeviceOptions_ExecutesRealInference()
    {
        var options = new MultiDeviceExecutorOptions()
            .AddOptions(ConfigureModelPath)
            .AddOptions(ConfigureModelPath);
        IAllocatingStep<Tensor<long>[], Tensor<float>[]> step =
            ModelExecutorFactory.CreateMaterializingModelStep(ModelExecutorType.Simple, options);

        await AssertFiniteStepOutput(step);
    }

    [Fact]
    public void CreateMaterializingModelStep_ShouldReturnSimpleStep_WhenOnnxOptionsProvided()
    {
        // Arrange
        var options = new OnnxModelExecutorOptions().ConfigureOnnxOptions(opt =>
        {
            opt.ModelDir = Path.GetDirectoryName(_modelPath)!;
            opt.ModelFileName = Path.GetFileName(_modelPath);
        });

        // Act
        var step = ModelExecutorFactory.CreateMaterializingModelStep(ModelExecutorType.Simple, options);

        // Assert
        Assert.IsType<OnnxModelExecutor>(step);
    }

    [Fact]
    public void CreateMaterializingModelStep_ShouldThrow_WhenUnknownExecutorType()
    {
        // Arrange
        var options = new OnnxModelExecutorOptions();
        var unknownType = (ModelExecutorType)999;

        // Act & Assert
        Assert.Throws<NotImplementedException>(() => ModelExecutorFactory.CreateMaterializingModelStep(unknownType, options));
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
        IAllocatingStep<Tensor<long>[], Tensor<float>[]> step)
    {
        Tensor<long>[] input = [Tensor.Create([11L, 22L, 33L], [1, 3])];
        using BatchLease<Tensor<float>[]> output =
            await step.RentOutputAsync(input, TestContext.Current.CancellationToken);

        await step.ExecuteAsync(input, output.Value, TestContext.Current.CancellationToken);

        Tensor<float> result = output.Value[0];
        Assert.Equal([11.0f, 22.0f, 33.0f], [result[0, 0], result[0, 1], result[0, 2]]);
    }
}
