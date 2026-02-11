using FAI.Core.Abstractions;
using FAI.Core.PipelineBatchExecutors;
using NSubstitute;

namespace FAI.Core.Tests.PipelineBatchExecutorTests;

public class SinkPipelineBatchExecutorTests
{
    [Fact]
    public async Task ExecuteBatchPredict_CallsInferenceSteps()
    {
        // Arrange
        var steps = Substitute.For<IInferenceSteps<int, int>>();
        var executor = new SinkPipelineBatchExecutor<int, int>(steps);
        int[] inputData = [1, 2, 3];
        var inputs = inputData.AsMemory();
        var outputs = new int[3].AsMemory();

        steps.ProcessBatch(Arg.Any<ReadOnlyMemory<int>>(), Arg.Any<Memory<int>>())
            .Returns(Task.CompletedTask);

        // Act
        await executor.ExecuteBatchPredict(inputs, outputs);

        // Assert
        await steps.Received(1).ProcessBatch(
            Arg.Is<ReadOnlyMemory<int>>(m => m.ToArray().SequenceEqual(inputData)),
            Arg.Any<Memory<int>>());
    }

    [Fact]
    public async Task ExecuteBatchPredict_HandlesEmptyInput()
    {
        // Arrange
        var steps = Substitute.For<IInferenceSteps<int, int>>();
        var executor = new SinkPipelineBatchExecutor<int, int>(steps);
        var inputs = ReadOnlyMemory<int>.Empty;
        var outputs = Memory<int>.Empty;

        // Act
        await executor.ExecuteBatchPredict(inputs, outputs);

        // Assert
        await steps.Received(1).ProcessBatch(Arg.Any<ReadOnlyMemory<int>>(), Arg.Any<Memory<int>>());
    }
}
