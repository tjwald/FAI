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

    [Fact]
    public async Task ExecuteBatchPredict_MultipleExecutors_RoutesCorrectly()
    {
        // Arrange
        var executor1 = Substitute.For<IPipelineBatchExecutor<int, int>>();
        var executor2 = Substitute.For<IPipelineBatchExecutor<int, int>>();
        var strategy = Substitute.For<IBatchExecutionRoutingStrategy<int, int>>();
        var routingExecutor = new RoutingPipelineBatchExecutor<int, int>([executor1, executor2], strategy);

        int[] inputsArray = [1, 2, 3, 4];
        var inputs = inputsArray.AsMemory();
        var outputs = new int[4].AsMemory();

        var route1 = new BatchExecutionRoutingResult<int, int>(executor1, [0..2]);
        var route2 = new BatchExecutionRoutingResult<int, int>(executor2, [2..4]);

        strategy.Route(Arg.Any<IPipelineBatchExecutor<int, int>[]>(), inputs)
            .Returns([route1, route2]);

        executor1.ExecuteBatchPredict(Arg.Any<ReadOnlyMemory<int>>(), Arg.Any<Memory<int>>())
            .Returns(async x =>
            {
                var input = (ReadOnlyMemory<int>)x[0];
                var output = (Memory<int>)x[1];
                for (int i = 0; i < input.Length; i++) output.Span[i] = input.Span[i] * 10;
                await Task.CompletedTask;
            });

        executor2.ExecuteBatchPredict(Arg.Any<ReadOnlyMemory<int>>(), Arg.Any<Memory<int>>())
            .Returns(async x =>
            {
                var input = (ReadOnlyMemory<int>)x[0];
                var output = (Memory<int>)x[1];
                for (int i = 0; i < input.Length; i++) output.Span[i] = input.Span[i] * 100;
                await Task.CompletedTask;
            });

        // Act
        await routingExecutor.ExecuteBatchPredict(inputs, outputs);

        // Assert
        int[] expected = [10, 20, 300, 400];
        Assert.Equal(expected, outputs.ToArray());
    }

    [Fact]
    public async Task ExecuteBatchPredict_EmptyResults_DoesNothing()
    {
        // Arrange
        var strategy = Substitute.For<IBatchExecutionRoutingStrategy<int, int>>();
        var routingExecutor = new RoutingPipelineBatchExecutor<int, int>([], strategy);
        var inputs = new int[5].AsMemory();
        var outputs = new int[5].AsMemory();

        strategy.Route(Arg.Any<IPipelineBatchExecutor<int, int>[]>(), inputs).Returns([]);

        // Act
        await routingExecutor.ExecuteBatchPredict(inputs, outputs);

        // Assert
        Assert.All(outputs.ToArray(), x => Assert.Equal(0, x));
    }

    [Fact]
    public async Task ExecuteBatchPredict_HandlesEmptyInput()
    {
        // Arrange
        var strategy = Substitute.For<IBatchExecutionRoutingStrategy<int, int>>();
        var routingExecutor = new RoutingPipelineBatchExecutor<int, int>([], strategy);
        var inputs = ReadOnlyMemory<int>.Empty;
        var outputs = Memory<int>.Empty;

        strategy.Route(Arg.Any<IPipelineBatchExecutor<int, int>[]>(), inputs).Returns([]);

        // Act
        await routingExecutor.ExecuteBatchPredict(inputs, outputs);

        // Assert
        // No exceptions should be thrown, and nothing should happen.
        strategy.Received(1).Route(Arg.Any<IPipelineBatchExecutor<int, int>[]>(), inputs);
    }
}
