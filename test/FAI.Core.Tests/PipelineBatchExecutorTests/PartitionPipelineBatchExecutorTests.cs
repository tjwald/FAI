using FAI.Core.Abstractions;
using FAI.Core.PipelineBatchExecutors;
using NSubstitute;

namespace FAI.Core.Tests.PipelineBatchExecutorTests;

public class PartitionPipelineBatchExecutorTests
{
    [Fact]
    public async Task ExecuteBatchPredict_SlicesAndSchedules()
    {
        // Arrange
        var schedular = Substitute.For<IBatchSchedular<int, int>>();
        var slicer = Substitute.For<IBatchSlicer<int>>();
        var innerExecutor = Substitute.For<IPipelineBatchExecutor<int, int>>();
        var executor = new PartitionPipelineBatchExecutor<int, int>(schedular, slicer, innerExecutor);

        int[] inputData = [1, 2, 3, 4];
        var inputs = inputData.AsMemory();
        var outputs = new int[4].AsMemory();
        Range[] ranges = [0..2, 2..4];

        slicer.Slice(inputs).Returns(ranges);
        schedular.RunInExecutor(innerExecutor, ranges, inputs, outputs).Returns(Task.CompletedTask);

        // Act
        await executor.ExecuteBatchPredict(inputs, outputs);

        // Assert
        slicer.Received(1).Slice(inputs);
        await schedular.Received(1).RunInExecutor(innerExecutor, ranges, inputs, outputs);
    }

    [Fact]
    public async Task ExecuteBatchPredict_HandlesEmptyInput()
    {
        // Arrange
        var schedular = Substitute.For<IBatchSchedular<int, int>>();
        var slicer = Substitute.For<IBatchSlicer<int>>();
        var innerExecutor = Substitute.For<IPipelineBatchExecutor<int, int>>();
        var executor = new PartitionPipelineBatchExecutor<int, int>(schedular, slicer, innerExecutor);

        var inputs = ReadOnlyMemory<int>.Empty;
        var outputs = Memory<int>.Empty;
        Range[] ranges = [];

        slicer.Slice(inputs).Returns(ranges);
        schedular.RunInExecutor(innerExecutor, ranges, inputs, outputs).Returns(Task.CompletedTask);

        // Act
        await executor.ExecuteBatchPredict(inputs, outputs);

        // Assert
        slicer.Received(1).Slice(inputs);
        await schedular.Received(1).RunInExecutor(innerExecutor, ranges, inputs, outputs);
    }
}
