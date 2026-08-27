using System.Numerics.Tensors;
using FAI.Core;
using FAI.Core.Steps;
using FAI.Onnx.Configuration;
using FAI.Onnx.ModelExecutors;

namespace FAI.Onnx.Tests.ModelExecutors;

public class AsyncOnnxModelExecutorTests(OnnxModelFixture fixture) : IClassFixture<OnnxModelFixture>
{
    private readonly string _modelPath = fixture.ModelPath;

    [Fact]
    public async Task ExecuteAsync_ShouldWriteCallerProvidedOutput()
    {
        var options = new OnnxModelExecutorOptions().ConfigureOnnxOptions(opt =>
        {
            opt.ModelDir = Path.GetDirectoryName(_modelPath)!;
            opt.ModelFileName = Path.GetFileName(_modelPath);
        });

        IAllocatingStep<Tensor<long>[], Tensor<float>[]> step = AsyncOnnxModelExecutor.FromPretrained(options);
        Tensor<long>[] inputs = [Tensor.Create([10L, 20L, 30L], [1, 3])];

        using BatchLease<Tensor<float>[]> output = await step.RentOutputAsync(inputs, TestContext.Current.CancellationToken);
        await step.ExecuteAsync(inputs, output.Value, TestContext.Current.CancellationToken);

        Assert.Equal(2, output.Value[0].Rank);
        Assert.Equal(1, output.Value[0].Lengths[0]);
        Assert.Equal(3, output.Value[0].Lengths[1]);
        Assert.Equal(10.0f, output.Value[0][0, 0]);
        Assert.Equal(20.0f, output.Value[0][0, 1]);
        Assert.Equal(30.0f, output.Value[0][0, 2]);
    }

    [Fact]
    public async Task ExecuteAsync_BorrowedOutputIsConsumedWithoutMaterialization()
    {
        var options = new OnnxModelExecutorOptions().ConfigureOnnxOptions(opt =>
        {
            opt.ModelDir = Path.GetDirectoryName(_modelPath)!;
            opt.ModelFileName = Path.GetFileName(_modelPath);
        });
        IBorrowedTensorProducer<Tensor<long>[], float> step = AsyncOnnxModelExecutor.FromPretrained(options);
        Tensor<long>[] inputs = [Tensor.Create([10L, 20L, 30L], [1, 3])];
        var output = new float[3];

        await step.ExecuteAsync(inputs, output, new CopyBorrowedOutput(), TestContext.Current.CancellationToken);

        Assert.Equal([10.0f, 20.0f, 30.0f], output);
    }

    private sealed class CopyBorrowedOutput : IBorrowedTensorConsumer<float, float[]>
    {
        public void Consume(ReadOnlyTensorSpan<float> tensor, int outputIndex, float[] output)
        {
            Assert.Equal(0, outputIndex);
            foreach (ReadOnlyTensorSpan<float> row in tensor.GetDimensionSpan(0))
            {
                row.AsSpan().CopyTo(output);
            }
        }
    }
}
