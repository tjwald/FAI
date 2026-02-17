using FAI.Core.Abstractions;
using FAI.Core.BatchSchedulers;

namespace FAI.Core.Tests.BatchSchedularTests;

public class SerialBatchSchedularTests
{
    [Fact]
    public async Task RunInExecutor_ExecutesAllRangesSequentially()
    {
        // Arrange
        var scheduler = new SerialBatchSchedular<int, int>();
        var executor = Substitute.For<IPipelineBatchExecutor<int, int>>();
        var ranges = new[] { new Range(0, 2), new Range(2, 5) };
        var inputs = new int[5].AsMemory();
        var outputs = new int[5].AsMemory();

        // Act
        await scheduler.RunInExecutor(executor, ranges, inputs, outputs);

        // Assert
        await executor.Received(1).ExecuteBatchPredict(Arg.Is<ReadOnlyMemory<int>>(m => m.Length == 2), Arg.Is<Memory<int>>(m => m.Length == 2));
        await executor.Received(1).ExecuteBatchPredict(Arg.Is<ReadOnlyMemory<int>>(m => m.Length == 3), Arg.Is<Memory<int>>(m => m.Length == 3));
    }
}
