using FAI.Core.Abstractions;
using FAI.Core.BatchSchedulers;
using FAI.Core.Configurations.PipelineBatchExecutors;
using NSubstitute;

namespace FAI.Core.Tests.BatchSchedularTests;

public class ParallelBatchSchedularTests
{
    [Fact]
    public async Task RunInExecutor_ExecutesAllRanges()
    {
        // Arrange
        var options = new ParallelBatchSchedularOptions { MaxConcurrency = 2 };
        var scheduler = new ParallelBatchSchedular<int, int>(options);
        var executor = Substitute.For<IPipelineBatchExecutor<int, int>>();
        Range[] ranges = [new Range(0, 2), new Range(2, 4), new Range(4, 5)];
        var inputs = new int[5].AsMemory();
        var outputs = new int[5].AsMemory();

        // Act
        await scheduler.RunInExecutor(executor, ranges, inputs, outputs);

        // Assert
        await executor.Received(3).ExecuteBatchPredict(Arg.Any<ReadOnlyMemory<int>>(), Arg.Any<Memory<int>>());
    }

    [Fact]
    public async Task RunInExecutor_RespectsConcurrencyLimit()
    {
        // Arrange
        var options = new ParallelBatchSchedularOptions { MaxConcurrency = 1 };
        var scheduler = new ParallelBatchSchedular<int, int>(options);
        var executor = Substitute.For<IPipelineBatchExecutor<int, int>>();

        int activeTasks = 0;
        int maxSeenActiveTasks = 0;
        var lockObj = new System.Threading.Lock();

        executor.ExecuteBatchPredict(Arg.Any<ReadOnlyMemory<int>>(), Arg.Any<Memory<int>>())
            .Returns(async _ =>
            {
                lock (lockObj)
                {
                    activeTasks++;
                    maxSeenActiveTasks = Math.Max(maxSeenActiveTasks, activeTasks);
                }
                await Task.Delay(10);
                lock (lockObj)
                {
                    activeTasks--;
                }
            });

        var ranges = Enumerable.Range(0, 10).Select(i => new Range(i, i + 1)).ToList();
        var inputs = new int[10].AsMemory();
        var outputs = new int[10].AsMemory();

        // Act
        await scheduler.RunInExecutor(executor, ranges, inputs, outputs);

        // Assert
        Assert.Equal(1, maxSeenActiveTasks);
    }

    [Fact]
    public async Task RunInExecutor_HandlesEmptyInputs()
    {
        // Arrange
        var options = new ParallelBatchSchedularOptions { MaxConcurrency = 2 };
        var scheduler = new ParallelBatchSchedular<int, int>(options);
        var executor = Substitute.For<IPipelineBatchExecutor<int, int>>();
        var ranges = Enumerable.Empty<Range>();
        var inputs = ReadOnlyMemory<int>.Empty;
        var outputs = Memory<int>.Empty;

        // Act
        await scheduler.RunInExecutor(executor, ranges, inputs, outputs);

        // Assert
        await executor.DidNotReceive().ExecuteBatchPredict(Arg.Any<ReadOnlyMemory<int>>(), Arg.Any<Memory<int>>());
    }
}
