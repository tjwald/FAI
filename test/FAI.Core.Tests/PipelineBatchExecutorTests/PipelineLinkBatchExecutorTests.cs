using System.Buffers;
using FAI.Core.Abstractions;
using FAI.Core.PipelineBatchExecutors;
using NSubstitute;

namespace FAI.Core.Tests.PipelineBatchExecutorTests;

public class PipelineLinkBatchExecutorTests
{
    [Fact]
    public async Task ExecuteBatchPredict_TransformsUserIdsToStringsForNextPipeline()
    {
        // Arrange: Transform user IDs (int) to strings for downstream text processing pipeline
        var nextPipeline = Substitute.For<IPipeline<string, int>>();
        Func<int, string> userIdToString = userId => $"USER_{userId:D6}"; // Format: USER_000123
        var pool = ArrayPool<string>.Shared;
        var executor = new PipelineLinkBatchExecutor<int, string, int>(nextPipeline, userIdToString, pool);

        int[] userIdsArray = [123, 456, 789];
        ReadOnlyMemory<int> userIds = userIdsArray;
        Memory<int> sentimentScores = new int[3];

        nextPipeline.BatchPredict(Arg.Any<ReadOnlyMemory<string>>(), Arg.Any<Memory<int>>())
            .Returns(async callInfo =>
            {
                var transformedInputs = callInfo.ArgAt<ReadOnlyMemory<string>>(0);
                var outputs = callInfo.ArgAt<Memory<int>>(1);

                // Simulate sentiment analysis: longer strings = higher scores
                for (int i = 0; i < transformedInputs.Length; i++)
                {
                    outputs.Span[i] = transformedInputs.Span[i].Length;
                }

                await Task.CompletedTask;
            });

        // Act
        await executor.ExecuteBatchPredict(userIds, sentimentScores);

        // Assert: Verify transformation was applied correctly
        string[] expectedTransformed = ["USER_000123", "USER_000456", "USER_000789"];
        await nextPipeline.Received(1).BatchPredict(
            Arg.Is<ReadOnlyMemory<string>>(m => m.ToArray().SequenceEqual(expectedTransformed)),
            Arg.Any<Memory<int>>());

        // Verify outputs were populated by next pipeline
        Assert.Equal(11, sentimentScores.Span[0]); // "USER_000123".Length = 11
        Assert.Equal(11, sentimentScores.Span[1]); // "USER_000456".Length = 11
        Assert.Equal(11, sentimentScores.Span[2]); // "USER_000789".Length = 11
    }

    [Fact]
    public async Task ExecuteBatchPredict_UsesArrayPoolForIntermediateBuffer()
    {
        // Arrange: Demonstrate zero-allocation pattern using ArrayPool
        var nextPipeline = Substitute.For<IPipeline<string, bool>>();
        Func<double, string> temperatureFormatter = temp => $"{temp:F1}°C";
        var pool = ArrayPool<string>.Shared;
        var executor = new PipelineLinkBatchExecutor<double, string, bool>(nextPipeline, temperatureFormatter, pool);

        double[] temperaturesArray = [36.6, 37.2, 38.5, 39.1];
        ReadOnlyMemory<double> temperatures = temperaturesArray;
        Memory<bool> isFever = new bool[4];

        nextPipeline.BatchPredict(Arg.Any<ReadOnlyMemory<string>>(), Arg.Any<Memory<bool>>())
            .Returns(async callInfo =>
            {
                var formattedTemps = callInfo.ArgAt<ReadOnlyMemory<string>>(0);
                var outputs = callInfo.ArgAt<Memory<bool>>(1);

                // Determine fever status: > 37.5°C
                for (int i = 0; i < formattedTemps.Length; i++)
                {
                    // Parse temperature from formatted string
                    string temp = formattedTemps.Span[i];
                    double value = double.Parse(temp.Replace("°C", ""));
                    outputs.Span[i] = value > 37.5;
                }

                await Task.CompletedTask;
            });

        // Act
        await executor.ExecuteBatchPredict(temperatures, isFever);

        // Assert: Verify correct transformation and results
        Assert.False(isFever.Span[0]); // 36.6 - normal
        Assert.False(isFever.Span[1]); // 37.2 - normal
        Assert.True(isFever.Span[2]);  // 38.5 - fever
        Assert.True(isFever.Span[3]);  // 39.1 - fever
    }
}
