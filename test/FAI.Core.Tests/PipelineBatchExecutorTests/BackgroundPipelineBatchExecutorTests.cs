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
        var next = Substitute.For<IPipelineBatchExecutor<int, int>>();
        var executor = new BackgroundPipelineBatchExecutor<int, int>(next, 1);
        var inputs = new int[5].AsMemory();
        var outputs = new int[5].AsMemory();

        next.ExecuteBatchPredict(Arg.Any<ReadOnlyMemory<int>>(), Arg.Any<Memory<int>>())
            .Returns(Task.CompletedTask);

        // Act
        await executor.ExecuteBatchPredict(inputs, outputs);

        // Assert
        await next.Received(1).ExecuteBatchPredict(Arg.Any<ReadOnlyMemory<int>>(), Arg.Any<Memory<int>>());
    }

    [Fact]
    public async Task ExecuteBatchPredict_PropagatesException()
    {
        // Arrange
        var next = Substitute.For<IPipelineBatchExecutor<int, int>>();
        var executor = new BackgroundPipelineBatchExecutor<int, int>(next, 1);
        var inputs = new int[5].AsMemory();
        var outputs = new int[5].AsMemory();

        var exception = new InvalidOperationException("Test exception");
        next.ExecuteBatchPredict(Arg.Any<ReadOnlyMemory<int>>(), Arg.Any<Memory<int>>())
            .Returns(Task.FromException(exception));

        // Act & Assert
        var ex = await Assert.ThrowsAsync<InvalidOperationException>(() => executor.ExecuteBatchPredict(inputs, outputs));
        Assert.Equal("Test exception", ex.Message);
    }

    [Fact]
    public async Task ExecuteBatchPredict_HandlesEmptyInput()
    {
        // Arrange
        var next = Substitute.For<IPipelineBatchExecutor<int, int>>();
        var executor = new BackgroundPipelineBatchExecutor<int, int>(next, 1);
        var inputs = ReadOnlyMemory<int>.Empty;
        var outputs = Memory<int>.Empty;

        next.ExecuteBatchPredict(Arg.Any<ReadOnlyMemory<int>>(), Arg.Any<Memory<int>>())
            .Returns(Task.CompletedTask);

        // Act
        await executor.ExecuteBatchPredict(inputs, outputs);

        // Assert
        await next.Received(1).ExecuteBatchPredict(Arg.Any<ReadOnlyMemory<int>>(), Arg.Any<Memory<int>>());
    }
}
