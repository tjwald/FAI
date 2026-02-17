using System.Numerics.Tensors;
using FAI.Core.Abstractions;
using FAI.Core.Configurations.InferenceTasks;
using FAI.Core.InferenceTasks.Classification;

namespace FAI.Core.Tests;

public class ClassificationTaskTests
{
    private class TestClassificationTask : ClassificationTask<string, Tensor<float>[], float, string, float>
    {
        public TestClassificationTask(
            IPreprocessor<string, Tensor<float>[], float> preprocessor,
            IModelExecutor<float, float> modelExecutor,
            ClassificationOptions<string> pipelineOptions)
            : base(preprocessor, modelExecutor, pipelineOptions)
        {
        }
    }

    private class MockModelExecutor : IModelExecutor<float, float>
    {
        public Task<Tensor<float>[]> RunAsync(Tensor<float>[] inputs) => throw new NotImplementedException();

        public Task RunAsync(Tensor<float>[] inputs, Action<ReadOnlyTensorSpan<float>, int> postProcess)
        {
            postProcess(new ReadOnlyTensorSpan<float>([1.0f, 2.0f, -1.0f, 5.0f], [2, 2]), 0);
            return Task.CompletedTask;
        }
    }

    [Fact]
    public async Task RunModel_CorrectlyProcessesLogits()
    {
        // Arrange
        string[] choices = ["Negative", "Positive"];
        var options = new ClassificationOptions<string>(choices);
        var preprocessor = Substitute.For<IPreprocessor<string, Tensor<float>[], float>>();
        var modelExecutor = new MockModelExecutor();

        var task = new TestClassificationTask(preprocessor, modelExecutor, options);

        string[] inputs = ["input1", "input2"];
        Tensor<float>[] tensors = [Tensor.Create<float>([2, 5])]; // 2 items, 5 features (but we only need 2 for choices)

        // Act
        var results = await task.RunModel(inputs.AsMemory(), tensors);

        // Assert
        Assert.Equal(2, results.Length);
        Assert.Equal("Positive", results[0].Choice);
        Assert.Equal("Positive", results[1].Choice);
        Assert.True(results[0].Score > 0.5f);
        Assert.True(results[1].Score > 0.5f);
    }

    [Fact]
    public void GetClassificationResult_ReturnsHighestProbability()
    {
        // Arrange
        string[] choices = ["A", "B", "C"];
        var options = new ClassificationOptions<string>(choices);
        float[] logits = [1.0f, 5.0f, 2.0f];

        // Act
        var result = options.GetClassificationResult(logits);

        // Assert
        Assert.Equal("B", result.Choice);
        Assert.True(result.Score > 0.9f); // Softmax of [1, 5, 2] should highly favor 5
    }
}
