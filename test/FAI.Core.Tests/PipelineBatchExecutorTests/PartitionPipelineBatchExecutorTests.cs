using FAI.Core.Abstractions;
using FAI.Core.PipelineBatchExecutors;
using NSubstitute;

namespace FAI.Core.Tests.PipelineBatchExecutorTests;

public class PartitionPipelineBatchExecutorTests
{
    [Fact]
    public async Task ExecuteBatchPredict_SlicesAndSchedules()
    {
        // Arrange - Demonstrate batch partitioning for parallel processing
        int[] inputData = [100, 200, 300, 400, 500, 600];
        ReadOnlyMemory<int> inputs = inputData;
        int[] outputData = new int[6];
        Memory<int> outputs = outputData;

        // Mock slicer to partition batch into smaller chunks
        var slicer = Substitute.For<IBatchSlicer<int>>();
        Range[] partitions = [0..2, 2..4, 4..6]; // 3 partitions of 2 items each
        slicer.Slice(Arg.Any<ReadOnlyMemory<int>>()).Returns(partitions);

        // Mock scheduler to coordinate parallel execution
        var schedular = Substitute.For<IBatchSchedular<int, int>>();
        schedular.RunInExecutor(
                Arg.Any<IPipelineBatchExecutor<int, int>>(),
                Arg.Any<IEnumerable<Range>>(),
                Arg.Any<ReadOnlyMemory<int>>(),
                Arg.Any<Memory<int>>())
            .Returns(callInfo =>
            {
                var executor = callInfo.ArgAt<IPipelineBatchExecutor<int, int>>(0);
                var ranges = callInfo.ArgAt<IEnumerable<Range>>(1);
                var inp = callInfo.ArgAt<ReadOnlyMemory<int>>(2);
                var outp = callInfo.ArgAt<Memory<int>>(3);

                // Simulate parallel processing of each partition
                foreach (var range in ranges)
                {
                    for (int i = range.Start.Value; i < range.End.Value; i++)
                    {
                        outp.Span[i] = inp.Span[i] / 10; // Divide by 10
                    }
                }
                return Task.CompletedTask;
            });

        var innerExecutor = Substitute.For<IPipelineBatchExecutor<int, int>>();
        var executor = new PartitionPipelineBatchExecutor<int, int>(schedular, slicer, innerExecutor);

        // Act
        await executor.ExecuteBatchPredict(inputs, outputs);

        // Assert
        slicer.Received(1).Slice(Arg.Any<ReadOnlyMemory<int>>());
        await schedular.Received(1).RunInExecutor(
            innerExecutor,
            Arg.Is<IEnumerable<Range>>(r => r.SequenceEqual(partitions)),
            Arg.Any<ReadOnlyMemory<int>>(),
            Arg.Any<Memory<int>>());

        Assert.Equal([10, 20, 30, 40, 50, 60], outputData);
    }

    [Fact]
    public async Task ExecuteBatchPredict_HandlesEmptyInput()
    {
        // Arrange
        ReadOnlyMemory<int> inputs = Array.Empty<int>();
        Memory<int> outputs = Array.Empty<int>();

        var slicer = Substitute.For<IBatchSlicer<int>>();
        Range[] emptyRanges = [];
        slicer.Slice(Arg.Any<ReadOnlyMemory<int>>()).Returns(emptyRanges);

        var schedular = Substitute.For<IBatchSchedular<int, int>>();
        schedular.RunInExecutor(
                Arg.Any<IPipelineBatchExecutor<int, int>>(),
                Arg.Any<IEnumerable<Range>>(),
                Arg.Any<ReadOnlyMemory<int>>(),
                Arg.Any<Memory<int>>())
            .Returns(Task.CompletedTask);

        var innerExecutor = Substitute.For<IPipelineBatchExecutor<int, int>>();
        var executor = new PartitionPipelineBatchExecutor<int, int>(schedular, slicer, innerExecutor);

        // Act
        await executor.ExecuteBatchPredict(inputs, outputs);

        // Assert
        slicer.Received(1).Slice(Arg.Any<ReadOnlyMemory<int>>());
        await schedular.Received(1).RunInExecutor(
            innerExecutor,
            Arg.Is<IEnumerable<Range>>(r => !r.Any()),
            Arg.Any<ReadOnlyMemory<int>>(),
            Arg.Any<Memory<int>>());
    }
}
