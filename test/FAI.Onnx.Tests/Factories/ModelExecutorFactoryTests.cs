using FAI.Core.Configurations.ModelExecutors;
using FAI.Core.ModelExecutors;
using FAI.Onnx.Configuration;
using FAI.Onnx.Factories;
using FAI.Onnx.ModelExecutors;

namespace FAI.Onnx.Tests.Factories;

public class ModelExecutorFactoryTests(OnnxModelFixture fixture) : IClassFixture<OnnxModelFixture>
{
    private readonly string _modelPath = fixture.ModelPath;

    [Fact]
    public void CreateModelExecutor_ShouldReturnPooledModelExecutor_WhenMultiDeviceOptionsProvided()
    {
        // Arrange
        var options = new MultiDeviceExecutorOptions()
            .AddOptions(opt => opt.ConfigureOnnxOptions(onnx =>
            {
                onnx.ModelDir = Path.GetDirectoryName(_modelPath)!;
                onnx.ModelFileName = Path.GetFileName(_modelPath);
            }));

        // Act
        var executor = ModelExecutorFactory.CreateModelExecutor(ModelExecutorType.Simple, options);

        // Assert
        Assert.IsType<PooledModelExecutor<long, float>>(executor);
    }

    [Fact]
    public void CreateModelExecutor_ShouldReturnPooledModelExecutor_WhenPooledOptionsProvided()
    {
        // Arrange
        var onnxOptions = new OnnxModelExecutorOptions().ConfigureOnnxOptions(opt =>
        {
            opt.ModelDir = Path.GetDirectoryName(_modelPath)!;
            opt.ModelFileName = Path.GetFileName(_modelPath);
        });

        var options = new PooledExecutorOptions<OnnxModelExecutorOptions>(onnxOptions, 2);

        // Act
        var executor = ModelExecutorFactory.CreateModelExecutor(ModelExecutorType.Async, options);

        // Assert
        Assert.IsType<PooledModelExecutor<long, float>>(executor);
    }

    [Fact]
    public void CreateModelExecutor_ShouldReturnSimpleExecutor_WhenOnnxModelExecutorOptionsProvided()
    {
        // Arrange
        var options = new OnnxModelExecutorOptions().ConfigureOnnxOptions(opt =>
        {
            opt.ModelDir = Path.GetDirectoryName(_modelPath)!;
            opt.ModelFileName = Path.GetFileName(_modelPath);
        });

        // Act
        var executor = ModelExecutorFactory.CreateModelExecutor(ModelExecutorType.Simple, options);

        // Assert
        Assert.IsType<OnnxModelExecutor>(executor);
    }

    [Fact]
    public void CreateModelExecutor_ShouldThrow_WhenUnknownExecutorType()
    {
        // Arrange
        var options = new OnnxModelExecutorOptions();
        var unknownType = (ModelExecutorType)999;

        // Act & Assert
        Assert.Throws<NotImplementedException>(() => ModelExecutorFactory.CreateModelExecutor(unknownType, options));
    }
}
