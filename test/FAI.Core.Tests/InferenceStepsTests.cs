using FAI.Core.Abstractions;
using NSubstitute;

namespace FAI.Core.Tests;

public class InferenceStepsTests
{
    private class TestInferenceSteps : InferenceSteps<string, int, double, string>
    {
        public override int Preprocess(ReadOnlySpan<string> input) => input.Length;

        public override Task<double> RunModel(ReadOnlyMemory<string> input, int preprocesses)
        {
            return Task.FromResult((double)preprocesses * 2.0);
        }

        public override void PostProcess(ReadOnlySpan<string> inputs, int preprocesses, double modelOutput, Span<string> outputs)
        {
            for (int i = 0; i < outputs.Length; i++)
            {
                outputs[i] = $"{modelOutput}";
            }
        }
    }

    [Fact]
    public async Task ProcessBatch_ExecutesAllStepsInOrder()
    {
        // Arrange
        var steps = new TestInferenceSteps();
        var inputs = new[] { "a", "b", "c" };
        var outputs = new string[3];

        // Act
        await steps.ProcessBatch(inputs, outputs);

        // Assert
        Assert.All(outputs, o => Assert.Equal("6", o));
    }

    private class MockInferenceSteps : TestInferenceSteps
    {
        public bool PreprocessCalled { get; private set; }
        public bool RunModelCalled { get; private set; }
        public bool PostProcessCalled { get; private set; }

        public override int Preprocess(ReadOnlySpan<string> input)
        {
            PreprocessCalled = true;
            return base.Preprocess(input);
        }

        public override Task<double> RunModel(ReadOnlyMemory<string> input, int preprocesses)
        {
            RunModelCalled = true;
            return base.RunModel(input, preprocesses);
        }

        public override void PostProcess(ReadOnlySpan<string> inputs, int preprocesses, double modelOutput, Span<string> outputs)
        {
            PostProcessCalled = true;
            base.PostProcess(inputs, preprocesses, modelOutput, outputs);
        }
    }

    [Fact]
    public async Task ProcessBatch_ExecutesAllSteps()
    {
        // Arrange
        var steps = new MockInferenceSteps();
        var inputs = new[] { "a", "b" }.AsMemory();
        var outputs = new string[2].AsMemory();

        // Act
        await steps.ProcessBatch(inputs, outputs);

        // Assert
        Assert.True(steps.PreprocessCalled);
        Assert.True(steps.RunModelCalled);
        Assert.True(steps.PostProcessCalled);
    }
}
