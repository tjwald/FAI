using FAI.Core.Abstractions;
using FAI.Core.PipelineBatchExecutors;

namespace FAI.Core.Tests.PipelineBatchExecutorTests;

public class SinkPipelineBatchExecutorTests
{
    [Fact]
    public async Task ExecuteBatchPredict_ProcessesImageBatchThroughInferenceSteps()
    {
        // Arrange: Sink executor terminates pipeline by invoking actual ML inference
        var inferenceSteps = Substitute.For<IInferenceSteps<string, int>>();
        var executor = new SinkPipelineBatchExecutor<string, int>(inferenceSteps);

        string[] imagePathsArray = ["cat.jpg", "dog.jpg", "bird.jpg"];
        ReadOnlyMemory<string> imagePaths = imagePathsArray.AsMemory();
        Memory<int> classifications = new int[3];

        inferenceSteps.ProcessBatch(Arg.Any<ReadOnlyMemory<string>>(), Arg.Any<Memory<int>>())
            .Returns(async callInfo =>
            {
                var inputs = callInfo.ArgAt<ReadOnlyMemory<string>>(0);
                var outputs = callInfo.ArgAt<Memory<int>>(1);

                // Simulate image classification: assign class IDs based on filename
                for (int i = 0; i < inputs.Length; i++)
                {
                    string filename = inputs.Span[i];
                    outputs.Span[i] = filename.Contains("cat") ? 0 :
                                     filename.Contains("dog") ? 1 : 2;
                }

                await Task.CompletedTask;
            });

        // Act
        await executor.ExecuteBatchPredict(imagePaths, classifications);

        // Assert: Verify inference steps were invoked and outputs populated
        string[] expectedPaths = ["cat.jpg", "dog.jpg", "bird.jpg"];
        await inferenceSteps.Received(1).ProcessBatch(
            Arg.Is<ReadOnlyMemory<string>>(m => m.ToArray().SequenceEqual(expectedPaths)),
            Arg.Any<Memory<int>>());

        Assert.Equal(0, classifications.Span[0]); // cat
        Assert.Equal(1, classifications.Span[1]); // dog
        Assert.Equal(2, classifications.Span[2]); // bird
    }

    [Fact]
    public async Task ExecuteBatchPredict_HandlesEmptyBatch()
    {
        // Arrange: Demonstrate graceful handling of empty batches
        var inferenceSteps = Substitute.For<IInferenceSteps<double, bool>>();
        var executor = new SinkPipelineBatchExecutor<double, bool>(inferenceSteps);

        var inputs = ReadOnlyMemory<double>.Empty;
        var outputs = Memory<bool>.Empty;

        // Act
        await executor.ExecuteBatchPredict(inputs, outputs);

        // Assert: Should still invoke inference steps (even with empty batch)
        await inferenceSteps.Received(1).ProcessBatch(
            Arg.Is<ReadOnlyMemory<double>>(m => m.Length == 0),
            Arg.Is<Memory<bool>>(m => m.Length == 0));
    }

    [Fact]
    public async Task ExecuteBatchPredict_PropagatesOutputsFromInferenceSteps()
    {
        // Arrange: Verify that outputs from inference steps are correctly propagated
        var inferenceSteps = Substitute.For<IInferenceSteps<float, float>>();
        var executor = new SinkPipelineBatchExecutor<float, float>(inferenceSteps);

        float[] sensorReadingsArray = [23.5f, 25.1f, 22.8f, 24.3f];
        ReadOnlyMemory<float> sensorReadings = sensorReadingsArray.AsMemory();
        Memory<float> normalizedValues = new float[4];

        inferenceSteps.ProcessBatch(Arg.Any<ReadOnlyMemory<float>>(), Arg.Any<Memory<float>>())
            .Returns(async callInfo =>
            {
                var inputs = callInfo.ArgAt<ReadOnlyMemory<float>>(0);
                var outputs = callInfo.ArgAt<Memory<float>>(1);

                // Normalize sensor readings to 0-1 range
                float min = 22.8f;
                float max = 25.1f;
                for (int i = 0; i < inputs.Length; i++)
                {
                    outputs.Span[i] = (inputs.Span[i] - min) / (max - min);
                }

                await Task.CompletedTask;
            });

        // Act
        await executor.ExecuteBatchPredict(sensorReadings, normalizedValues);

        // Assert: Verify normalized outputs
        Assert.Equal(0.304f, normalizedValues.Span[0], precision: 3); // (23.5-22.8)/(25.1-22.8)
        Assert.Equal(1.000f, normalizedValues.Span[1], precision: 3); // (25.1-22.8)/(25.1-22.8)
        Assert.Equal(0.000f, normalizedValues.Span[2], precision: 3); // (22.8-22.8)/(25.1-22.8)
        Assert.Equal(0.652f, normalizedValues.Span[3], precision: 3); // (24.3-22.8)/(25.1-22.8)
    }
}
