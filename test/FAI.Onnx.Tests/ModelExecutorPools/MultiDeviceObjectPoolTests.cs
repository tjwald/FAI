using FAI.Onnx.Configuration;
using FAI.Onnx.ModelExecutorPools;
using FAI.Onnx.ModelExecutors;

namespace FAI.Onnx.Tests.ModelExecutorPools;

public class MultiDeviceObjectPoolTests(OnnxModelFixture fixture) : IClassFixture<OnnxModelFixture>
{
    private readonly string _modelPath = fixture.ModelPath;

    [Fact]
    public void Get_ShouldReturnExecutorsInRoundRobinOrder()
    {
        // Arrange
        var options = new OnnxModelExecutorOptions().ConfigureOnnxOptions(opt =>
        {
            opt.ModelDir = Path.GetDirectoryName(_modelPath)!;
            opt.ModelFileName = Path.GetFileName(_modelPath);
        });

        var exec1 = OnnxModelExecutor.FromPretrained(options);
        var exec2 = OnnxModelExecutor.FromPretrained(options);
        var executors = new List<OnnxModelExecutorBase> { exec1, exec2 };
        var pool = new MultiDeviceObjectPool(executors);

        // Act & Assert
        Assert.Same(exec1, pool.Get());
        Assert.Same(exec2, pool.Get());
        Assert.Same(exec1, pool.Get());
    }
}
