using FAI.Core.Abstractions;
using FAI.Core.PipelineBatchExecutors;
using NSubstitute;

namespace FAI.Core.Tests.PipelineBatchExecutorTests;

public class RoutingPipelineBatchExecutorTests
{
    [Fact]
    public async Task ExecuteBatchPredict_RoutesCorrectlyAndMergesOutputs()
    {
        // Arrange
        var executor1 = Substitute.For<IPipelineBatchExecutor<int, int>>();
        var strategy = Substitute.For<IBatchExecutionRoutingStrategy<int, int>>();
        var routingExecutor = new RoutingPipelineBatchExecutor<int, int>([executor1], strategy);

        var inputs = new int[] { 10, 20, 30, 40, 50 }.AsMemory();
        var outputs = new int[5].AsMemory();

        // Route: items at index 0, 2, 4 to executor1
        var ranges = new List<Range> { new Range(0, 1), new Range(2, 3), new Range(4, 5) };
        var routingResult = new BatchExecutionRoutingResult<int, int>(executor1, ranges);
        strategy.Route(Arg.Any<IPipelineBatchExecutor<int, int>[]>(), inputs)
            .Returns([routingResult]);

        executor1.ExecuteBatchPredict(Arg.Any<ReadOnlyMemory<int>>(), Arg.Any<Memory<int>>())
            .Returns(async x =>
            {
                var input = (ReadOnlyMemory<int>)x[0];
                var output = (Memory<int>)x[1];
                for (int i = 0; i < input.Length; i++) output.Span[i] = input.Span[i] + 1;
                await Task.CompletedTask;
            });

        // Act
        await routingExecutor.ExecuteBatchPredict(inputs, outputs);

        // Assert
        Assert.Equal(11, outputs.Span[0]); // 10 + 1
        Assert.Equal(0, outputs.Span[1]);  // Not routed
        Assert.Equal(31, outputs.Span[2]); // 30 + 1
        Assert.Equal(0, outputs.Span[3]);  // Not routed
        Assert.Equal(51, outputs.Span[4]); // 50 + 1
    }
}
