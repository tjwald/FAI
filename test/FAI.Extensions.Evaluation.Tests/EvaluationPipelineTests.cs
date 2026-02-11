using FAI.Core.Abstractions;
using Microsoft.Extensions.Logging.Abstractions;
using NSubstitute;

namespace FAI.Extensions.Evaluation.Tests;

public class EvaluationPipelineTests
{
    public record TestLoaderInput(int Count);
    public record TestInferenceInput(int Id);
    public record TestLoadedInput(int Id) : IInferenceInputGetter<TestInferenceInput>
    {
        public TestInferenceInput InferenceInput => new(Id);
    }
    public record TestInferenceOutput(int Id, string Prediction);
    public record TestEvaluationResult(double Accuracy);

    private static async IAsyncEnumerable<T> ToAsync<T>(IEnumerable<T> items)
    {
        foreach (var item in items)
        {
            yield return item;
            await Task.Yield();
        }
    }

    [Fact]
    public async Task Evaluate_SimpleFlow_ShouldReturnCorrectResults()
    {
        // Arrange
        var dataLoader = Substitute.For<IDataLoader<TestLoaderInput, TestLoadedInput, TestInferenceInput>>();
        var inference = Substitute.For<IInference<TestInferenceInput, TestInferenceOutput>>();
        var evaluator = Substitute.For<IEvaluator<TestLoadedInput, TestInferenceOutput, TestEvaluationResult>>();
        var logger = NullLogger<EvaluationPipeline<TestLoaderInput, TestLoadedInput, TestInferenceInput, TestInferenceOutput, TestEvaluationResult>>.Instance;
        var options = new EvaluationPipelineOptions();

        var pipeline = new EvaluationPipeline<TestLoaderInput, TestLoadedInput, TestInferenceInput, TestInferenceOutput, TestEvaluationResult>(
            dataLoader, inference, evaluator, logger, options);

        var inputs = Enumerable.Range(0, 5).Select(i => new TestLoadedInput(i)).ToArray();
        dataLoader.LoadData(Arg.Any<TestLoaderInput>()).Returns(ToAsync(inputs));

        inference.BatchPredict(Arg.Any<ReadOnlyMemory<TestInferenceInput>>())
            .Returns(args => Task.FromResult(((ReadOnlyMemory<TestInferenceInput>)args[0]).ToArray().Select(x => new TestInferenceOutput(x.Id, "ok")).ToArray()));

        evaluator.Evaluate(Arg.Any<IAsyncEnumerable<(TestLoadedInput[], TestInferenceOutput[])>>())
            .Returns(async args =>
            {
                var stream = (IAsyncEnumerable<(TestLoadedInput[], TestInferenceOutput[])>)args[0];
                await foreach (var _ in stream) { }
                return new TestEvaluationResult(1.0);
            });

        // Act
        var result = await pipeline.Evaluate(new TestLoaderInput(5));

        // Assert
        Assert.Equal(5, result.SampleSize);
        Assert.Equal(1.0, result.Evaluation.Accuracy);
        dataLoader.Received(1).LoadData(Arg.Is<TestLoaderInput>(x => x.Count == 5));
        await inference.Received(1).BatchPredict(Arg.Any<ReadOnlyMemory<TestInferenceInput>>());
    }

    [Fact]
    public async Task Evaluate_WithChunking_ShouldCallInferenceMultipleTimes()
    {
        // Arrange
        var dataLoader = Substitute.For<IDataLoader<TestLoaderInput, TestLoadedInput, TestInferenceInput>>();
        var inference = Substitute.For<IInference<TestInferenceInput, TestInferenceOutput>>();
        var evaluator = Substitute.For<IEvaluator<TestLoadedInput, TestInferenceOutput, TestEvaluationResult>>();
        var logger = NullLogger<EvaluationPipeline<TestLoaderInput, TestLoadedInput, TestInferenceInput, TestInferenceOutput, TestEvaluationResult>>.Instance;
        var options = new EvaluationPipelineOptions(LoadingChunkSize: 2);

        var pipeline = new EvaluationPipeline<TestLoaderInput, TestLoadedInput, TestInferenceInput, TestInferenceOutput, TestEvaluationResult>(
            dataLoader, inference, evaluator, logger, options);

        var inputs = Enumerable.Range(0, 5).Select(i => new TestLoadedInput(i)).ToArray();
        dataLoader.LoadData(Arg.Any<TestLoaderInput>()).Returns(ToAsync(inputs));

        inference.BatchPredict(Arg.Any<ReadOnlyMemory<TestInferenceInput>>())
            .Returns(args => Task.FromResult(((ReadOnlyMemory<TestInferenceInput>)args[0]).ToArray().Select(x => new TestInferenceOutput(x.Id, "ok")).ToArray()));

        evaluator.Evaluate(Arg.Any<IAsyncEnumerable<(TestLoadedInput[], TestInferenceOutput[])>>())
            .Returns(async args =>
            {
                var stream = (IAsyncEnumerable<(TestLoadedInput[], TestInferenceOutput[])>)args[0];
                await foreach (var _ in stream) { }
                return new TestEvaluationResult(0.8);
            });

        // Act
        var result = await pipeline.Evaluate(new TestLoaderInput(5));

        // Assert
        Assert.Equal(5, result.SampleSize);
        // Expect 2 calls of 2 elements and 1 call of 1 element
        await inference.Received(3).BatchPredict(Arg.Any<ReadOnlyMemory<TestInferenceInput>>());
    }

    [Fact]
    public async Task Evaluate_WithParallelOptions_ShouldExecuteSuccessfully()
    {
        // Arrange
        var dataLoader = Substitute.For<IDataLoader<TestLoaderInput, TestLoadedInput, TestInferenceInput>>();
        var inference = Substitute.For<IInference<TestInferenceInput, TestInferenceOutput>>();
        var evaluator = Substitute.For<IEvaluator<TestLoadedInput, TestInferenceOutput, TestEvaluationResult>>();
        var logger = NullLogger<EvaluationPipeline<TestLoaderInput, TestLoadedInput, TestInferenceInput, TestInferenceOutput, TestEvaluationResult>>.Instance;
        var options = new EvaluationPipelineOptions(LoadingChunkSize: 2, ParallelLoading: true, ParallelEvaluation: true);

        var pipeline = new EvaluationPipeline<TestLoaderInput, TestLoadedInput, TestInferenceInput, TestInferenceOutput, TestEvaluationResult>(
            dataLoader, inference, evaluator, logger, options);

        var inputs = Enumerable.Range(0, 4).Select(i => new TestLoadedInput(i)).ToArray();
        dataLoader.LoadData(Arg.Any<TestLoaderInput>()).Returns(ToAsync(inputs));

        inference.BatchPredict(Arg.Any<ReadOnlyMemory<TestInferenceInput>>())
            .Returns(args => Task.FromResult(((ReadOnlyMemory<TestInferenceInput>)args[0]).ToArray().Select(x => new TestInferenceOutput(x.Id, "ok")).ToArray()));

        evaluator.Evaluate(Arg.Any<IAsyncEnumerable<(TestLoadedInput[], TestInferenceOutput[])>>())
            .Returns(async args =>
            {
                var stream = (IAsyncEnumerable<(TestLoadedInput[], TestInferenceOutput[])>)args[0];
                await foreach (var _ in stream) { }
                return new TestEvaluationResult(1.0);
            });

        // Act
        var result = await pipeline.Evaluate(new TestLoaderInput(4));

        // Assert
        Assert.Equal(4, result.SampleSize);
        await evaluator.Received(1).Evaluate(Arg.Any<IAsyncEnumerable<(TestLoadedInput[], TestInferenceOutput[])>>());
    }
}
