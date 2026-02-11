using FAI.Core.Abstractions;
using FAI.Core.Pipelines;
using Microsoft.Extensions.Logging.Abstractions;
using NSubstitute;

namespace FAI.Core.Tests.PipelineTests;

public class AccumulatingPipelineTests
{
    [Fact]
    public async Task Predict_MaxBatchSizeTrigger_FlushesBatch()
    {
        // Arrange
        var executor = Substitute.For<IPipelineBatchExecutor<int, int>>();
        var options = new AccumulatingPipelineOptions
        {
            MaxBatchSize = 3,
            MaxLatency = TimeSpan.FromSeconds(10)
        };
        var policy = Substitute.For<IFailedBatchPolicy<int, int>>();
        var pipeline = new AccumulatingPipeline<int, int>(executor, options, policy, NullLogger<AccumulatingPipeline<int, int>>.Instance);

        executor.ExecuteBatchPredict(Arg.Any<ReadOnlyMemory<int>>(), Arg.Any<Memory<int>>())
            .Returns(async x =>
            {
                var input = (ReadOnlyMemory<int>)x[0];
                var output = (Memory<int>)x[1];
                for (int i = 0; i < input.Length; i++) output.Span[i] = input.Span[i] * 2;
                await Task.CompletedTask;
            });

        // Act
        var t1 = pipeline.Predict(1);
        var t2 = pipeline.Predict(2);
        var t3 = pipeline.Predict(3);

        var results = await Task.WhenAll(t1, t2, t3);

        // Assert
        Assert.Equal(new[] { 2, 4, 6 }, results);
        await executor.Received(1).ExecuteBatchPredict(Arg.Is<ReadOnlyMemory<int>>(m => m.Length == 3), Arg.Any<Memory<int>>());
    }

    [Fact]
    public async Task Predict_MaxLatencyTrigger_FlushesPartialBatch()
    {
        // Arrange
        var executor = Substitute.For<IPipelineBatchExecutor<int, int>>();
        var options = new AccumulatingPipelineOptions
        {
            MaxBatchSize = 10,
            MaxLatency = TimeSpan.FromMilliseconds(50)
        };
        var policy = Substitute.For<IFailedBatchPolicy<int, int>>();
        var pipeline = new AccumulatingPipeline<int, int>(executor, options, policy, NullLogger<AccumulatingPipeline<int, int>>.Instance);

        executor.ExecuteBatchPredict(Arg.Any<ReadOnlyMemory<int>>(), Arg.Any<Memory<int>>())
            .Returns(async x =>
            {
                var input = (ReadOnlyMemory<int>)x[0];
                var output = (Memory<int>)x[1];
                for (int i = 0; i < input.Length; i++) output.Span[i] = input.Span[i] * 2;
                await Task.CompletedTask;
            });

        // Act
        var result = await pipeline.Predict(5);

        // Assert
        Assert.Equal(10, result);
        await executor.Received(1).ExecuteBatchPredict(Arg.Is<ReadOnlyMemory<int>>(m => m.Length == 1), Arg.Any<Memory<int>>());
    }

    [Fact]
    public async Task Predict_FailedBatch_InvokesPolicy()
    {
        // Arrange
        var executor = Substitute.For<IPipelineBatchExecutor<int, int>>();
        var options = new AccumulatingPipelineOptions
        {
            MaxBatchSize = 1,
            MaxLatency = TimeSpan.FromSeconds(1)
        };
        var policy = Substitute.For<IFailedBatchPolicy<int, int>>();
        var pipeline = new AccumulatingPipeline<int, int>(executor, options, policy, NullLogger<AccumulatingPipeline<int, int>>.Instance);

        var exception = new Exception("Fail");
        executor.ExecuteBatchPredict(Arg.Any<ReadOnlyMemory<int>>(), Arg.Any<Memory<int>>())
            .Returns(Task.FromException(exception));

        policy.HandleAsync(Arg.Any<ReadOnlyMemory<int>>(), Arg.Any<Memory<int>>(), executor, exception, Arg.Any<CancellationToken>())
            .Returns(x =>
            {
                var output = (Memory<int>)x[1];
                output.Span[0] = -1; // Fallback
                return Task.CompletedTask;
            });

        // Act
        var result = await pipeline.Predict(100);

        // Assert
        Assert.Equal(-1, result);
        await policy.Received(1).HandleAsync(Arg.Any<ReadOnlyMemory<int>>(), Arg.Any<Memory<int>>(), executor, exception, Arg.Any<CancellationToken>());
    }

    [Fact]
    public async Task Predict_Disposed_ThrowsObjectDisposedException()
    {
        // Arrange
        var executor = Substitute.For<IPipelineBatchExecutor<int, int>>();
        var options = new AccumulatingPipelineOptions
        {
            MaxBatchSize = 10,
            MaxLatency = TimeSpan.FromSeconds(10)
        };
        var policy = Substitute.For<IFailedBatchPolicy<int, int>>();
        var pipeline = new AccumulatingPipeline<int, int>(executor, options, policy, NullLogger<AccumulatingPipeline<int, int>>.Instance);

        pipeline.Dispose();

        // Act & Assert
        await Assert.ThrowsAsync<ObjectDisposedException>(() => pipeline.Predict(2));
    }

    [Fact]
    public async Task BatchPredict_EmptyInput_ReturnsEmptyResults()
    {
        // Arrange
        var executor = Substitute.For<IPipelineBatchExecutor<int, int>>();
        var options = new AccumulatingPipelineOptions { MaxBatchSize = 1 };
        var pipeline = new AccumulatingPipeline<int, int>(executor, options, Substitute.For<IFailedBatchPolicy<int, int>>(), NullLogger<AccumulatingPipeline<int, int>>.Instance);

        // Act
        var results = await pipeline.BatchPredict(ReadOnlyMemory<int>.Empty);

        // Assert
        Assert.Empty(results);
    }
}
