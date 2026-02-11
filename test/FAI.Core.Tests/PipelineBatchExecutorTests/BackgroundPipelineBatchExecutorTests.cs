using FAI.Core.Abstractions;
using FAI.Core.PipelineBatchExecutors;
using NSubstitute;

namespace FAI.Core.Tests.PipelineBatchExecutorTests;

public class BackgroundPipelineBatchExecutorTests
{
    [Fact]
    public async Task ExecuteBatchPredict_OffloadsToWorkerAndPropagatesResult()
    {
        // Arrange
        int[] inputData = [10, 20, 30, 40, 50];
        ReadOnlyMemory<int> inputs = inputData;
        int[] outputArray = new int[5];
        Memory<int> outputs = outputArray;

        var next = Substitute.For<IPipelineBatchExecutor<int, int>>();
        next.ExecuteBatchPredict(Arg.Any<ReadOnlyMemory<int>>(), Arg.Any<Memory<int>>())
            .Returns(callInfo =>
            {
                var input = callInfo.ArgAt<ReadOnlyMemory<int>>(0);
                var output = callInfo.ArgAt<Memory<int>>(1);

                // Simulate actual processing: double each input value
                for (int i = 0; i < input.Length; i++)
                {
                    output.Span[i] = input.Span[i] * 2;
                }
                return Task.CompletedTask;
            });

        var executor = new BackgroundPipelineBatchExecutor<int, int>(next, workerCount: 1);

        // Act
        await executor.ExecuteBatchPredict(inputs, outputs);

        // Assert
        await next.Received(1).ExecuteBatchPredict(
            Arg.Is<ReadOnlyMemory<int>>(m => m.Length == 5),
            Arg.Any<Memory<int>>());

        Assert.Equal([20, 40, 60, 80, 100], outputArray);
    }

    [Fact]
    public async Task ExecuteBatchPredict_PropagatesException()
    {
        // Arrange
        int[] inputData = [1, 2, 3];
        ReadOnlyMemory<int> inputs = inputData;
        Memory<int> outputs = new int[3];

        var next = Substitute.For<IPipelineBatchExecutor<int, int>>();
        var exception = new InvalidOperationException("Model execution failed");
        next.ExecuteBatchPredict(Arg.Any<ReadOnlyMemory<int>>(), Arg.Any<Memory<int>>())
            .Returns(Task.FromException(exception));

        var executor = new BackgroundPipelineBatchExecutor<int, int>(next, workerCount: 2);

        // Act & Assert
        var ex = await Assert.ThrowsAsync<InvalidOperationException>(() =>
            executor.ExecuteBatchPredict(inputs, outputs));
        Assert.Equal("Model execution failed", ex.Message);
    }

    [Fact]
    public async Task ExecuteBatchPredict_HandlesEmptyInput()
    {
        // Arrange
        ReadOnlyMemory<int> inputs = Array.Empty<int>();
        Memory<int> outputs = Array.Empty<int>();

        var next = Substitute.For<IPipelineBatchExecutor<int, int>>();
        next.ExecuteBatchPredict(Arg.Any<ReadOnlyMemory<int>>(), Arg.Any<Memory<int>>())
            .Returns(Task.CompletedTask);

        var executor = new BackgroundPipelineBatchExecutor<int, int>(next, workerCount: 1);

        // Act
        await executor.ExecuteBatchPredict(inputs, outputs);

        // Assert
        await next.Received(1).ExecuteBatchPredict(
            Arg.Is<ReadOnlyMemory<int>>(m => m.Length == 0),
            Arg.Is<Memory<int>>(m => m.Length == 0));
    }
}
