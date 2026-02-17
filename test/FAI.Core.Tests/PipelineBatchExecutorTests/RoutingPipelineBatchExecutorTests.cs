using FAI.Core.Abstractions;
using FAI.Core.PipelineBatchExecutors;

namespace FAI.Core.Tests.PipelineBatchExecutorTests;

public class RoutingPipelineBatchExecutorTests
{
    [Fact]
    public async Task ExecuteBatchPredict_RoutesCorrectlyAndMergesOutputs()
    {
        // Arrange - Simulate routing specific batch items to specialized executors
        int[] inputData = [10, 20, 30, 40, 50];
        ReadOnlyMemory<int> inputs = inputData;
        int[] outputData = new int[5];
        Memory<int> outputs = outputData;

        var executor1 = Substitute.For<IPipelineBatchExecutor<int, int>>();
        executor1.ExecuteBatchPredict(Arg.Any<ReadOnlyMemory<int>>(), Arg.Any<Memory<int>>())
            .Returns(callInfo =>
            {
                var input = callInfo.ArgAt<ReadOnlyMemory<int>>(0);
                var output = callInfo.ArgAt<Memory<int>>(1);

                // Process by incrementing
                for (int i = 0; i < input.Length; i++)
                {
                    output.Span[i] = input.Span[i] + 1;
                }
                return Task.CompletedTask;
            });

        var strategy = Substitute.For<IBatchExecutionRoutingStrategy<int, int>>();
        // Route items at index 0, 2, 4 to executor1 (odd positions remain unprocessed)
        var routingResult = new BatchExecutionRoutingResult<int, int>(executor1, [0..1, 2..3, 4..5]);
        strategy.Route(Arg.Any<IPipelineBatchExecutor<int, int>[]>(), Arg.Any<ReadOnlyMemory<int>>())
            .Returns([routingResult]);

        var routingExecutor = new RoutingPipelineBatchExecutor<int, int>([executor1], strategy);

        // Act
        await routingExecutor.ExecuteBatchPredict(inputs, outputs);

        // Assert
        Assert.Equal(11, outputData[0]); // 10 + 1 (routed)
        Assert.Equal(0, outputData[1]);  // Not routed
        Assert.Equal(31, outputData[2]); // 30 + 1 (routed)
        Assert.Equal(0, outputData[3]);  // Not routed
        Assert.Equal(51, outputData[4]); // 50 + 1 (routed)
    }

    [Fact]
    public async Task ExecuteBatchPredict_MultipleExecutors_RoutesCorrectly()
    {
        // Arrange - Demonstrate routing to different executors (e.g., CPU vs GPU models)
        int[] inputData = [1, 2, 3, 4];
        ReadOnlyMemory<int> inputs = inputData;
        int[] outputData = new int[4];
        Memory<int> outputs = outputData;

        // Fast executor (CPU) - multiplies by 10
        var fastExecutor = Substitute.For<IPipelineBatchExecutor<int, int>>();
        fastExecutor.ExecuteBatchPredict(Arg.Any<ReadOnlyMemory<int>>(), Arg.Any<Memory<int>>())
            .Returns(callInfo =>
            {
                var input = callInfo.ArgAt<ReadOnlyMemory<int>>(0);
                var output = callInfo.ArgAt<Memory<int>>(1);
                for (int i = 0; i < input.Length; i++)
                {
                    output.Span[i] = input.Span[i] * 10;
                }
                return Task.CompletedTask;
            });

        // Accurate executor (GPU) - multiplies by 100
        var accurateExecutor = Substitute.For<IPipelineBatchExecutor<int, int>>();
        accurateExecutor.ExecuteBatchPredict(Arg.Any<ReadOnlyMemory<int>>(), Arg.Any<Memory<int>>())
            .Returns(callInfo =>
            {
                var input = callInfo.ArgAt<ReadOnlyMemory<int>>(0);
                var output = callInfo.ArgAt<Memory<int>>(1);
                for (int i = 0; i < input.Length; i++)
                {
                    output.Span[i] = input.Span[i] * 100;
                }
                return Task.CompletedTask;
            });

        var strategy = Substitute.For<IBatchExecutionRoutingStrategy<int, int>>();
        // Route first 2 items to fast executor, last 2 to accurate executor
        var route1 = new BatchExecutionRoutingResult<int, int>(fastExecutor, [0..2]);
        var route2 = new BatchExecutionRoutingResult<int, int>(accurateExecutor, [2..4]);
        strategy.Route(Arg.Any<IPipelineBatchExecutor<int, int>[]>(), Arg.Any<ReadOnlyMemory<int>>())
            .Returns([route1, route2]);

        var routingExecutor = new RoutingPipelineBatchExecutor<int, int>([fastExecutor, accurateExecutor], strategy);

        // Act
        await routingExecutor.ExecuteBatchPredict(inputs, outputs);

        // Assert
        Assert.Equal([10, 20, 300, 400], outputData);
    }

    [Fact]
    public async Task ExecuteBatchPredict_EmptyResults_DoesNothing()
    {
        // Arrange - No routing strategy returns empty results
        int[] inputData = [1, 2, 3, 4, 5];
        ReadOnlyMemory<int> inputs = inputData;
        int[] outputData = new int[5];
        Memory<int> outputs = outputData;

        var strategy = Substitute.For<IBatchExecutionRoutingStrategy<int, int>>();
        strategy.Route(Arg.Any<IPipelineBatchExecutor<int, int>[]>(), Arg.Any<ReadOnlyMemory<int>>())
            .Returns([]);

        var routingExecutor = new RoutingPipelineBatchExecutor<int, int>([], strategy);

        // Act
        await routingExecutor.ExecuteBatchPredict(inputs, outputs);

        // Assert - All outputs remain at default (0)
        Assert.All(outputData, x => Assert.Equal(0, x));
    }

    [Fact]
    public async Task ExecuteBatchPredict_HandlesEmptyInput()
    {
        // Arrange
        ReadOnlyMemory<int> inputs = Array.Empty<int>();
        Memory<int> outputs = Array.Empty<int>();

        var strategy = Substitute.For<IBatchExecutionRoutingStrategy<int, int>>();
        strategy.Route(Arg.Any<IPipelineBatchExecutor<int, int>[]>(), Arg.Any<ReadOnlyMemory<int>>())
            .Returns([]);

        var routingExecutor = new RoutingPipelineBatchExecutor<int, int>([], strategy);

        // Act
        await routingExecutor.ExecuteBatchPredict(inputs, outputs);

        // Assert
        strategy.Received(1).Route(Arg.Any<IPipelineBatchExecutor<int, int>[]>(), Arg.Any<ReadOnlyMemory<int>>());
    }
}
