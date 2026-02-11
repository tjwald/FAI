using FAI.Core.Abstractions;
using FAI.Core.Extensions.DI;
using Microsoft.Extensions.DependencyInjection;

namespace FAI.Core.Extensions.DI.Tests;

public class FaiBuilderExtensionsTests
{
    private readonly IServiceCollection _services = new ServiceCollection();

    [Fact]
    public void AddPipeline_RegistersPipelineAsSingleton()
    {
        // Arrange
        _services.AddSingleton<IInferenceSteps<string, int>, MockInferenceSteps>();

        // Act
        _services.AddPipeline<string, int>();
        var sp = _services.BuildServiceProvider();

        // Assert
        var pipeline = sp.GetService<IPipeline<string, int>>();
        Assert.NotNull(pipeline);

        var secondResolve = sp.GetService<IPipeline<string, int>>();
        Assert.Same(pipeline, secondResolve);
    }

    [Fact]
    public async Task UsePartitioning_AssemblesCorrectExecutor()
    {
        // Arrange
        _services.AddSingleton<IInferenceSteps<string, int>, MockInferenceSteps>();
        var tracker = new List<string>();
        _services.AddSingleton(tracker);
        var mockSlicer = Substitute.For<IBatchSlicer<string>>();
        mockSlicer.Slice(Arg.Any<ReadOnlyMemory<string>>()).Returns([new Range(0, 1)]);

        var mockSchedular = Substitute.For<IBatchSchedular<string, int>>();
        mockSchedular.RunInExecutor(Arg.Any<IPipelineBatchExecutor<string, int>>(), Arg.Any<IEnumerable<Range>>(), Arg.Any<ReadOnlyMemory<string>>(), Arg.Any<Memory<int>>())
            .Returns(Task.CompletedTask);

        // Act
        _services.AddPipeline<string, int>()
            .UsePartitioning(p =>
            {
                p.WithSlicer(_ => mockSlicer);
                p.WithSchedular(_ => mockSchedular);
            });

        var sp = _services.BuildServiceProvider();
        var pipeline = sp.GetRequiredService<IPipeline<string, int>>();

        var output = new int[1];
        await pipeline.BatchPredict(new string[] { "test" }.AsMemory(), output);

        // Assert
        await mockSchedular.Received(1).RunInExecutor(
            Arg.Any<IPipelineBatchExecutor<string, int>>(),
            Arg.Any<IEnumerable<Range>>(),
            Arg.Any<ReadOnlyMemory<string>>(),
            Arg.Any<Memory<int>>());
    }

    private class MockInferenceSteps : IInferenceSteps<string, int>
    {
        public Task ProcessBatch(ReadOnlyMemory<string> inputs, Memory<int> outputs) => Task.CompletedTask;
    }
}
