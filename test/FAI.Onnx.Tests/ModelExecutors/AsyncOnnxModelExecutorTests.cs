using System.Numerics.Tensors;
using FAI.Core;
using FAI.Core.Pipelines;
using FAI.Onnx.Configuration;
using FAI.Onnx.ModelExecutors;

namespace FAI.Onnx.Tests.ModelExecutors;

public class AsyncOnnxModelExecutorTests(OnnxModelFixture fixture) : IClassFixture<OnnxModelFixture>
{
    private readonly string _modelPath = fixture.ModelPath;

    [Fact]
    public async Task ExecuteAsync_ReturnsLiveTensorOutputs()
    {
        var options = new OnnxModelExecutorOptions().ConfigureOnnxOptions(opt =>
        {
            opt.ModelDir = Path.GetDirectoryName(_modelPath)!;
            opt.ModelFileName = Path.GetFileName(_modelPath);
        });

        IPipeline<Tensor<long>[], TensorOutputs<float>> pipeline = AsyncOnnxModelExecutor.FromPretrained(options);
        Tensor<long>[] inputs = [Tensor.Create([10L, 20L, 30L], [1, 3])];

        using TensorOutputs<float> output = await pipeline.ExecuteAsync(inputs, TestContext.Current.CancellationToken);
        ReadOnlyTensorSpan<float> tensor = output.GetOutput(0);

        Assert.Equal(2, tensor.Rank);
        Assert.Equal(1, tensor.Lengths[0]);
        Assert.Equal(3, tensor.Lengths[1]);
        Assert.Equal([10.0f, 20.0f, 30.0f], tensor.AsSpan().ToArray());
    }

    [Fact]
    public async Task ExecuteAsync_WithAlreadyCanceledToken_ThrowsOperationCanceledException()
    {
        var options = new OnnxModelExecutorOptions().ConfigureOnnxOptions(opt =>
        {
            opt.ModelDir = Path.GetDirectoryName(_modelPath)!;
            opt.ModelFileName = Path.GetFileName(_modelPath);
        });

        IPipeline<Tensor<long>[], TensorOutputs<float>> pipeline = AsyncOnnxModelExecutor.FromPretrained(options);
        Tensor<long>[] inputs = [Tensor.Create([10L, 20L, 30L], [1, 3])];

        using var cts = new CancellationTokenSource();
        cts.Cancel();

        await Assert.ThrowsAnyAsync<OperationCanceledException>(
            () => pipeline.ExecuteAsync(inputs, cts.Token).AsTask());
    }

    [Fact]
    public async Task ExecuteAsync_WhenCanceledWhileWaitingOnSemaphore_CancelsPromptly()
    {
        var options = new OnnxModelExecutorOptions().ConfigureOnnxOptions(opt =>
        {
            opt.ModelDir = Path.GetDirectoryName(_modelPath)!;
            opt.ModelFileName = Path.GetFileName(_modelPath);
        });

        // AsyncOnnxModelExecutor uses maxThreads = 1
        IPipeline<Tensor<long>[], TensorOutputs<float>> pipeline = AsyncOnnxModelExecutor.FromPretrained(options);
        Tensor<long>[] inputs = [Tensor.Create([10L, 20L, 30L], [1, 3])];

        using var cts = new CancellationTokenSource();
        cts.Cancel();

        // Second call with canceled token should cancel without waiting
        await Assert.ThrowsAnyAsync<OperationCanceledException>(
            () => pipeline.ExecuteAsync(inputs, cts.Token).AsTask());
    }
}
