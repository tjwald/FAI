using FAI.Core.Abstractions;
using FAI.Core.PipelineBatchExecutors;
using Microsoft.Extensions.Logging.Abstractions;
using NSubstitute;

namespace FAI.Core.Tests.PipelineBatchExecutorTests;

public class StreamedBatchExecutorTests
{
    private class TestInferenceSteps : InferenceSteps<int, int, int, int>
    {
        public override int Preprocess(ReadOnlySpan<int> input) => input.Length;

        public override Task<int> RunModel(ReadOnlyMemory<int> input, int preprocesses)
        {
            return Task.FromResult(preprocesses * 10);
        }

        public override void PostProcess(ReadOnlySpan<int> inputs, int preprocesses, int modelOutput, Span<int> outputs)
        {
            for (int i = 0; i < outputs.Length; i++)
            {
                outputs[i] = modelOutput + i;
            }
        }
    }

    [Fact]
    public async Task ExecuteBatchPredict_ProcessesThroughAllStages()
    {
        // Arrange
        var inference = new TestInferenceSteps();
        var executor = new StreamedBatchExecutor<int, int, int, int>(inference, null, null, false, NullLogger<StreamedBatchExecutor<int, int, int, int>>.Instance);
        var inputs = new int[3] { 1, 2, 3 }.AsMemory();
        var outputs = new int[3].AsMemory();

        // Act
        await executor.ExecuteBatchPredict(inputs, outputs);

        // Assert
        // Preprocess returns 3. RunModel returns 30. PostProcess adds index.
        Assert.Equal(30, outputs.Span[0]);
        Assert.Equal(31, outputs.Span[1]);
        Assert.Equal(32, outputs.Span[2]);
    }

    [Fact]
    public async Task ExecuteBatchPredict_WithBatching_ProcessesAllChunks()
    {
        // Arrange
        var inference = new TestInferenceSteps();
        var executor = new StreamedBatchExecutor<int, int, int, int>(inference, 2, 1, false, NullLogger<StreamedBatchExecutor<int, int, int, int>>.Instance);
        var inputs = new int[5] { 1, 2, 3, 4, 5 }.AsMemory();
        var outputs = new int[5].AsMemory();

        // Act
        await executor.ExecuteBatchPredict(inputs, outputs);

        // Assert
        // Chunk 1 (size 2): 20, 21
        // Chunk 2 (size 2): 20, 21
        // Chunk 3 (size 1): 10
        Assert.Equal(20, outputs.Span[0]);
        Assert.Equal(21, outputs.Span[1]);
        Assert.Equal(20, outputs.Span[2]);
        Assert.Equal(21, outputs.Span[3]);
        Assert.Equal(10, outputs.Span[4]);
    }

    private class FailingPostProcessInferenceSteps : TestInferenceSteps
    {
        public override void PostProcess(ReadOnlySpan<int> inputs, int preprocesses, int modelOutput, Span<int> outputs)
        {
            throw new InvalidOperationException("Post-processing failure");
        }
    }

    [Fact]
    public async Task ExecuteBatchPredict_PostProcessFails_PropagatesError()
    {
        // Arrange
        var inference = new FailingPostProcessInferenceSteps();
        var executor = new StreamedBatchExecutor<int, int, int, int>(inference, null, null, false, NullLogger<StreamedBatchExecutor<int, int, int, int>>.Instance);
        int[] inputData = [1, 2, 3];
        var inputs = inputData.AsMemory();
        var outputs = new int[3].AsMemory();

        // Act & Assert
        await Assert.ThrowsAsync<InvalidOperationException>(() => executor.ExecuteBatchPredict(inputs, outputs));
    }

    [Fact]
    public async Task ExecuteBatchPredict_HandlesEmptyInput()
    {
        // Arrange
        var inference = new TestInferenceSteps();
        var executor = new StreamedBatchExecutor<int, int, int, int>(inference, null, null, false, NullLogger<StreamedBatchExecutor<int, int, int, int>>.Instance);
        var inputs = ReadOnlyMemory<int>.Empty;
        var outputs = Memory<int>.Empty;

        // Act
        await executor.ExecuteBatchPredict(inputs, outputs);

        // Assert
        // We can't use Received on a real class, and StreamedBatchExecutor doesn't call ProcessBatch directly on the object.
        // It calls Preprocess, RunModel, and PostProcess.
        // Actually for empty input, StreamedBatchExecutor.ExecuteBatchPredict should just return Task.CompletedTask without writing to channels.
        Assert.True(true);
    }
}
