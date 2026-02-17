using FAI.Core.Abstractions;
using Microsoft.Extensions.DependencyInjection;

namespace FAI.Core.Extensions.DI.Tests;

public class PipelineBuilderTests
{
    private readonly IServiceCollection _services = new ServiceCollection();

    [Fact]
    public void Build_WithNoMiddleware_ReturnsPipelineWithDefaultSink()
    {
        // Arrange
        _services.AddSingleton<IInferenceSteps<string, int>, MockInferenceSteps>();
        var builder = new PipelineBuilder<string, int>(_services);

        // Act
        var pipeline = builder.Build(_services.BuildServiceProvider());

        // Assert
        Assert.NotNull(pipeline);
    }

    [Fact]
    public async Task Build_MaintainsExpectedChainOrder()
    {
        // Arrange
        var tracker = new List<string>();
        _services.AddSingleton(tracker);

        var builder = new PipelineBuilder<string, int>(_services);
        builder.Use((next, sp) => new OrderTrackingExecutor(next, "First", sp.GetRequiredService<List<string>>()));
        builder.Use((next, sp) => new OrderTrackingExecutor(next, "Second", sp.GetRequiredService<List<string>>()));
        builder.UseSink<OrderTrackingSink>(sp => new OrderTrackingSink("Sink", sp.GetRequiredService<List<string>>()));

        var sp = _services.BuildServiceProvider();

        // Act
        var pipeline = builder.Build(sp);
        var outputs = new int[1];
        string[] inputs = ["test"];
        await pipeline.BatchPredict(inputs, outputs);

        // Assert
        // Logic in PipelineBuilder: for (int i = _batchExecutorFactories.Count - 1; i >= 0; i--)
        // So the LAST added executor is the one wrapping the sink or the previous one?
        // Actually, the loop wraps 'current' (initially sink) with executors from last to first.
        // If i = 1 (Second), it wraps Sink. current = Second(Sink)
        // If i = 0 (First), it wraps current. current = First(Second(Sink))
        // So the order should be First -> Second -> Sink.
        Assert.Equal(["First", "Second", "Sink"], tracker);
    }

    [Fact]
    public void AddInferenceSteps_RegistersService()
    {
        // Arrange
        var builder = new PipelineBuilder<string, int>(_services);

        // Act
        builder.AddInferenceSteps<MockInferenceSteps>();

        // Assert
        var descriptor = _services.FirstOrDefault(d => d.ServiceType == typeof(IInferenceSteps<string, int>));
        Assert.NotNull(descriptor);
        Assert.Equal(typeof(MockInferenceSteps), descriptor.ImplementationType);
    }

    [Fact]
    public void AddModelExecutor_RegistersFactory()
    {
        // Arrange
        var builder = new PipelineBuilder<string, int>(_services);
        var mockExecutor = Substitute.For<IModelExecutor<string, int>>();

        // Act
        builder.AddModelExecutor(_ => mockExecutor);

        // Assert
        var sp = _services.BuildServiceProvider();
        var resolved = sp.GetService<IModelExecutor<string, int>>();
        Assert.Same(mockExecutor, resolved);
    }

    private class MockInferenceSteps : IInferenceSteps<string, int>
    {
        public Task ProcessBatch(ReadOnlyMemory<string> inputs, Memory<int> outputs) => Task.CompletedTask;
    }

    private class OrderTrackingExecutor(IPipelineBatchExecutor<string, int> next, string name, List<string> tracker) : IPipelineBatchExecutor<string, int>
    {
        public async Task ExecuteBatchPredict(ReadOnlyMemory<string> inputs, Memory<int> outputSpan)
        {
            tracker.Add(name);
            await next.ExecuteBatchPredict(inputs, outputSpan);
        }
    }

    private class OrderTrackingSink(string name, List<string> tracker) : IPipelineBatchExecutor<string, int>
    {
        public Task ExecuteBatchPredict(ReadOnlyMemory<string> inputs, Memory<int> outputSpan)
        {
            tracker.Add(name);
            return Task.CompletedTask;
        }
    }
}
