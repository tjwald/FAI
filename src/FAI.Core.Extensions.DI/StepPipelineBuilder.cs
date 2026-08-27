using System.Numerics;
using FAI.Core.Steps;

namespace FAI.Core.Extensions.DI;

public sealed class PipelineBuilder<TInput>
{
    private readonly IServiceCollection _services;

    internal PipelineBuilder(IServiceCollection services)
    {
        _services = services;
    }

    public ComposedPipelineBuilder<TInput, TOutput> Then<TOutput, TStep>(
        Action<PipelineStageBuilder<TInput, TOutput>>? configure = null)
        where TStep : class, IAllocatingStep<TInput, TOutput>
    {
        _services.AddSingleton<TStep>();
        return Then(serviceProvider => serviceProvider.GetRequiredService<TStep>(), configure);
    }

    public ComposedPipelineBuilder<TInput, TOutput> Then<TOutput>(
        Func<IServiceProvider, IAllocatingStep<TInput, TOutput>> stepFactory,
        Action<PipelineStageBuilder<TInput, TOutput>>? configure = null)
    {
        var stage = new PipelineStageBuilder<TInput, TOutput>();
        configure?.Invoke(stage);

        return new ComposedPipelineBuilder<TInput, TOutput>(
            _services,
            serviceProvider => new StepChain<TInput, TOutput>(
                stage.Build(serviceProvider, stepFactory(serviceProvider))));
    }

    public ComposedPipelineBuilder<TInput, TOutput> Then<TOutput>(
        Func<PipelineBuilder<TInput>, ComposedPipelineBuilder<TInput, TOutput>> buildPipeline,
        Func<IServiceProvider, TInput, CancellationToken, ValueTask<BatchLease<TOutput>>> rentOutput,
        Action<PipelineStageBuilder<TInput, TOutput>>? configure = null)
    {
        ComposedPipelineBuilder<TInput, TOutput> pipeline = buildPipeline(new PipelineBuilder<TInput>(_services));
        return Then(
            serviceProvider => new NestedPipelineStep<TInput, TOutput>(pipeline.BuildChain(serviceProvider), serviceProvider, rentOutput),
            configure);
    }

    public ComposedPipelineBuilder<TInput, TOutput> ThenBorrowed<TElement, TOutput>(
        Func<IServiceProvider, IBorrowedTensorProducer<TInput, TElement>> producerFactory,
        Func<IServiceProvider, IBorrowedTensorConsumer<TElement, TOutput>> consumerFactory,
        Func<IServiceProvider, TInput, CancellationToken, ValueTask<BatchLease<TOutput>>> rentOutput,
        Action<PipelineStageBuilder<TInput, TOutput>>? configure = null)
        where TElement : unmanaged, INumber<TElement>
    {
        return Then(
            serviceProvider => new BorrowedTensorDecodingStep<TInput, TElement, TOutput>(
                producerFactory(serviceProvider),
                consumerFactory(serviceProvider),
                (input, cancellationToken) => rentOutput(serviceProvider, input, cancellationToken)),
            configure);
    }
}

public sealed class ComposedPipelineBuilder<TStart, TCurrent>
{
    private readonly IServiceCollection _services;
    private readonly Func<IServiceProvider, IStepChain<TStart, TCurrent>> _build;

    internal ComposedPipelineBuilder(
        IServiceCollection services,
        Func<IServiceProvider, IStepChain<TStart, TCurrent>> build)
    {
        _services = services;
        _build = build;
    }

    public ComposedPipelineBuilder<TStart, TNext> Then<TNext, TStep>(
        Action<PipelineStageBuilder<TCurrent, TNext>>? configure = null)
        where TStep : class, IAllocatingStep<TCurrent, TNext>
    {
        _services.AddSingleton<TStep>();
        return Then(serviceProvider => serviceProvider.GetRequiredService<TStep>(), configure);
    }

    public ComposedPipelineBuilder<TStart, TNext> Then<TNext>(
        Func<IServiceProvider, IAllocatingStep<TCurrent, TNext>> stepFactory,
        Action<PipelineStageBuilder<TCurrent, TNext>>? configure = null)
    {
        var stage = new PipelineStageBuilder<TCurrent, TNext>();
        configure?.Invoke(stage);

        return new ComposedPipelineBuilder<TStart, TNext>(
            _services,
            serviceProvider => new AppendedStepChain<TStart, TCurrent, TNext>(
                _build(serviceProvider),
                stage.Build(serviceProvider, stepFactory(serviceProvider))));
    }

    public ComposedPipelineBuilder<TStart, TNext> Then<TNext>(
        Func<PipelineBuilder<TCurrent>, ComposedPipelineBuilder<TCurrent, TNext>> buildPipeline,
        Func<IServiceProvider, TCurrent, CancellationToken, ValueTask<BatchLease<TNext>>> rentOutput,
        Action<PipelineStageBuilder<TCurrent, TNext>>? configure = null)
    {
        ComposedPipelineBuilder<TCurrent, TNext> pipeline = buildPipeline(new PipelineBuilder<TCurrent>(_services));
        return Then(
            serviceProvider => new NestedPipelineStep<TCurrent, TNext>(pipeline.BuildChain(serviceProvider), serviceProvider, rentOutput),
            configure);
    }

    public ComposedPipelineBuilder<TStart, TNext> ThenBorrowed<TElement, TNext>(
        Func<IServiceProvider, IBorrowedTensorProducer<TCurrent, TElement>> producerFactory,
        Func<IServiceProvider, IBorrowedTensorConsumer<TElement, TNext>> consumerFactory,
        Func<IServiceProvider, TCurrent, CancellationToken, ValueTask<BatchLease<TNext>>> rentOutput,
        Action<PipelineStageBuilder<TCurrent, TNext>>? configure = null)
        where TElement : unmanaged, INumber<TElement>
    {
        return Then(
            serviceProvider => new BorrowedTensorDecodingStep<TCurrent, TElement, TNext>(
                producerFactory(serviceProvider),
                consumerFactory(serviceProvider),
                (input, cancellationToken) => rentOutput(serviceProvider, input, cancellationToken)),
            configure);
    }

    public IServiceCollection Build(string? key = null)
    {
        if (key is null)
        {
            _services.AddSingleton<IStep<TStart, TCurrent>>(serviceProvider => _build(serviceProvider));
        }
        else
        {
            _services.AddKeyedSingleton<IStep<TStart, TCurrent>>(key, (serviceProvider, _) => _build(serviceProvider));
        }

        return _services;
    }

    internal IStepChain<TStart, TCurrent> BuildChain(IServiceProvider serviceProvider) => _build(serviceProvider);
}

internal sealed class NestedPipelineStep<TInput, TOutput> : IAllocatingStep<TInput, TOutput>
{
    private readonly IStep<TInput, TOutput> _pipeline;
    private readonly IServiceProvider _serviceProvider;
    private readonly Func<IServiceProvider, TInput, CancellationToken, ValueTask<BatchLease<TOutput>>> _rentOutput;

    public NestedPipelineStep(
        IStep<TInput, TOutput> pipeline,
        IServiceProvider serviceProvider,
        Func<IServiceProvider, TInput, CancellationToken, ValueTask<BatchLease<TOutput>>> rentOutput)
    {
        _pipeline = pipeline;
        _serviceProvider = serviceProvider;
        _rentOutput = rentOutput;
    }

    public ValueTask<BatchLease<TOutput>> RentOutputAsync(
        TInput input,
        CancellationToken cancellationToken = default)
        => _rentOutput(_serviceProvider, input, cancellationToken);

    public ValueTask ExecuteAsync(TInput input, TOutput output, CancellationToken cancellationToken = default)
        => _pipeline.ExecuteAsync(input, output, cancellationToken);
}

public sealed class PipelineStageBuilder<TInput, TOutput>
{
    private readonly List<Func<IServiceProvider, IAllocatingStep<TInput, TOutput>, IAllocatingStep<TInput, TOutput>>> _decorators = [];

    public PipelineStageBuilder<TInput, TOutput> Use<TDecorator>()
        where TDecorator : class, IAllocatingStep<TInput, TOutput>
    {
        return Use((serviceProvider, inner) => ActivatorUtilities.CreateInstance<TDecorator>(serviceProvider, inner));
    }

    public PipelineStageBuilder<TInput, TOutput> Use(
        Func<IServiceProvider, IAllocatingStep<TInput, TOutput>, IAllocatingStep<TInput, TOutput>> factory)
    {
        _decorators.Add(factory);
        return this;
    }

    public PipelineStageBuilder<TInput, TOutput> UseBatchPartitioning<TInputBatch, TOutputBatch>()
        where TInputBatch : IReadOnlyIndexedBatch<TInput, TInputBatch>
        where TOutputBatch : IWritableIndexedBatch<TOutput, TOutputBatch>
    {
        return Use((serviceProvider, inner) =>
            new PartitioningStep<TInput, TOutput, TInputBatch, TOutputBatch>(
                inner,
                serviceProvider.GetRequiredService<IBatchPartitioner<TInput>>(),
                serviceProvider.GetService<IPartitionScheduler>()));
    }

    public PipelineStageBuilder<TInput, TOutput> UseOrdering<TInputBatch, TOutputBatch>()
        where TInputBatch : IReadOnlyIndexedBatch<TInput, TInputBatch>
        where TOutputBatch : IWritableIndexedBatch<TOutput, TOutputBatch>
    {
        return Use((serviceProvider, inner) =>
            new OrderingStep<TInput, TOutput, TInputBatch, TOutputBatch>(
                inner,
                serviceProvider.GetRequiredService<IIndexOrdering<TInput>>()));
    }

    internal IAllocatingStep<TInput, TOutput> Build(
        IServiceProvider serviceProvider,
        IAllocatingStep<TInput, TOutput> step)
    {
        IAllocatingStep<TInput, TOutput> current = step;
        for (int i = _decorators.Count - 1; i >= 0; i--)
        {
            current = _decorators[i](serviceProvider, current);
        }

        return current;
    }
}
