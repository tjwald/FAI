using System.Numerics.Tensors;
using FAI.Onnx.Configuration;
using FAI.Onnx.ModelExecutors;

namespace FAI.Onnx.Tests.ModelExecutors;

public class AsyncOnnxModelExecutorTests(OnnxModelFixture fixture) : IClassFixture<OnnxModelFixture>
{
    private readonly string _modelPath = fixture.ModelPath;

    [Fact]
    public async Task RunAsync_ShouldExecuteRealInference()
    {
        // Arrange
        var options = new OnnxModelExecutorOptions().ConfigureOnnxOptions(opt =>
        {
            opt.ModelDir = Path.GetDirectoryName(_modelPath)!;
            opt.ModelFileName = Path.GetFileName(_modelPath);
        });

        var executor = AsyncOnnxModelExecutor.FromPretrained(options);

        // Input matching the minimal model: [1, 3] of long
        var inputs = new[] { Tensor.Create([10L, 20L, 30L], [1, 3]) };

        // Act
        var results = await executor.RunAsync(inputs);

        // Assert
        Assert.Single(results);
        var output = results[0];
        Assert.Equal(2, output.Lengths.Length);
        Assert.Equal(1L, output.Lengths[0]);
        Assert.Equal(3L, output.Lengths[1]);

        // The minimal model casts long to float
        Assert.Equal(10.0f, output[0, 0]);
        Assert.Equal(20.0f, output[0, 1]);
        Assert.Equal(30.0f, output[0, 2]);
    }

    [Fact]
    public async Task RunAsync_WithPostProcess_ShouldExecuteRealInference()
    {
        // Arrange
        var options = new OnnxModelExecutorOptions().ConfigureOnnxOptions(opt =>
        {
            opt.ModelDir = Path.GetDirectoryName(_modelPath)!;
            opt.ModelFileName = Path.GetFileName(_modelPath);
        });

        var executor = AsyncOnnxModelExecutor.FromPretrained(options);
        var inputs = new[] { Tensor.Create([100L, 200L, 300L], [1, 3]) };

        var called = false;
        long[] outputShape = [];

        // Act
        await executor.RunAsync(inputs, (span, index) =>
        {
            called = true;
            outputShape = [(span.Lengths[0]), (span.Lengths[1])];
            Assert.Equal(100.0f, span[0, 0]);
            Assert.Equal(200.0f, span[0, 1]);
            Assert.Equal(300.0f, span[0, 2]);
        });

        // Assert
        Assert.True(called);
        Assert.Equal([1L, 3L], outputShape);
    }
}
