using FAI.Core.Steps;

namespace FAI.Core.Extensions.DI;

public delegate bool TryAllocatePipelineOutput<in TInput, TOutput>(TInput input, out TOutput output);

public sealed class PipelineBuilder<TInput>
{
    private readonly IServiceCollection _services;

    internal PipelineBuilder(IServiceCollection services)
    {
        _services = services;
    }

    public ComposedPipelineBuilder<TInput, TOutput> Then<TOutput, TStep>(
        Action<PipelineStageBuilder<TInput, TOutput>>? configure = null)
        where TStep : class, IStep<TInput, TOutput>
    {
        _services.AddSingleton<TStep>();
        return Then(serviceProvider => serviceProvider.GetRequiredService<TStep>(), configure);
    }

    public ComposedPipelineBuilder<TInput, TOutput> Then<TOutput>(
        Func<IServiceProvider, IStep<TInput, TOutput>> stepFactory,
        Action<PipelineStageBuilder<TInput, TOutput>>? configure = null)
    {
        var stage = new PipelineStageBuilder<TInput, TOutput>();
        configure?.Invoke(stage);

        return new ComposedPipelineBuilder<TInput, TOutput>(
            _services,
            serviceProvider => StepChain.Create(
                stage.Build(serviceProvider, stepFactory(serviceProvider))));
    }

    public ComposedPipelineBuilder<TInput, TOutput> Then<TOutput>(
        Func<PipelineBuilder<TInput>, ComposedPipelineBuilder<TInput, TOutput>> buildPipeline,
        Action<PipelineStageBuilder<TInput, TOutput>>? configure = null)
    {
        ComposedPipelineBuilder<TInput, TOutput> pipeline = buildPipeline(new PipelineBuilder<TInput>(_services));
        return Then(
            serviceProvider => pipeline.BuildChain(serviceProvider),
            configure);
    }

    public ComposedPipelineBuilder<TInput, TOutput> Then<TOutput>(
        Func<PipelineBuilder<TInput>, ComposedPipelineBuilder<TInput, TOutput>> buildPipeline,
        TryAllocatePipelineOutput<TInput, TOutput> tryAllocateOutput,
        Action<PipelineStageBuilder<TInput, TOutput>>? configure = null)
    {
        ComposedPipelineBuilder<TInput, TOutput> pipeline = buildPipeline(new PipelineBuilder<TInput>(_services));
        return Then(
            serviceProvider => new PreallocatingPipelineStep<TInput, TOutput>(
                pipeline.BuildChain(serviceProvider),
                tryAllocateOutput),
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
        where TStep : class, IStep<TCurrent, TNext>
    {
        _services.AddSingleton<TStep>();
        return Then(serviceProvider => serviceProvider.GetRequiredService<TStep>(), configure);
    }

    public ComposedPipelineBuilder<TStart, TNext> Then<TNext>(
        Func<IServiceProvider, IStep<TCurrent, TNext>> stepFactory,
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
        Action<PipelineStageBuilder<TCurrent, TNext>>? configure = null)
    {
        ComposedPipelineBuilder<TCurrent, TNext> pipeline = buildPipeline(new PipelineBuilder<TCurrent>(_services));
        return Then(
            serviceProvider => pipeline.BuildChain(serviceProvider),
            configure);
    }

    public ComposedPipelineBuilder<TStart, TNext> Then<TNext>(
        Func<PipelineBuilder<TCurrent>, ComposedPipelineBuilder<TCurrent, TNext>> buildPipeline,
        TryAllocatePipelineOutput<TCurrent, TNext> tryAllocateOutput,
        Action<PipelineStageBuilder<TCurrent, TNext>>? configure = null)
    {
        ComposedPipelineBuilder<TCurrent, TNext> pipeline = buildPipeline(new PipelineBuilder<TCurrent>(_services));
        return Then(
            serviceProvider => new PreallocatingPipelineStep<TCurrent, TNext>(
                pipeline.BuildChain(serviceProvider),
                tryAllocateOutput),
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

internal sealed class PreallocatingPipelineStep<TInput, TOutput> : IPreallocatingStep<TInput, TOutput>
{
    private readonly IStepChain<TInput, TOutput> _pipeline;
    private readonly TryAllocatePipelineOutput<TInput, TOutput> _tryAllocateOutput;

    public PreallocatingPipelineStep(
        IStepChain<TInput, TOutput> pipeline,
        TryAllocatePipelineOutput<TInput, TOutput> tryAllocateOutput)
    {
        if (!pipeline.CanWriteOutput)
        {
            throw new InvalidOperationException("A preallocating nested pipeline requires a destination-writing final stage.");
        }

        _pipeline = pipeline;
        _tryAllocateOutput = tryAllocateOutput;
    }

    public bool TryAllocateOutput(TInput input, out TOutput output)
        => _tryAllocateOutput(input, out output);

    public ValueTask<TOutput> ExecuteAsync(TInput input, CancellationToken cancellationToken = default)
        => _pipeline.ExecuteAsync(input, cancellationToken);

    public ValueTask ExecuteAsync(
        TInput input,
        TOutput output,
        CancellationToken cancellationToken = default)
        => _pipeline.ExecuteIntoAsync(input, output, cancellationToken);
}

public sealed class PipelineStageBuilder<TInput, TOutput>
{
    private readonly List<Func<IServiceProvider, IStep<TInput, TOutput>, IStep<TInput, TOutput>>> _decorators = [];

    public PipelineStageBuilder<TInput, TOutput> Use<TDecorator>()
        where TDecorator : class, IStep<TInput, TOutput>
    {
        return Use((serviceProvider, inner) => ActivatorUtilities.CreateInstance<TDecorator>(serviceProvider, inner));
    }

    public PipelineStageBuilder<TInput, TOutput> Use(
        Func<IServiceProvider, IStep<TInput, TOutput>, IStep<TInput, TOutput>> factory)
    {
        _decorators.Add(factory);
        return this;
    }

    public PipelineStageBuilder<TInput, TOutput> UseBatchPartitioning(
        IReadOnlyIndexedBatch<TInput> inputBatch,
        IWritableIndexedBatch<TOutput> outputBatch)
    {
        return Use((serviceProvider, inner) =>
            new PartitioningStep<TInput, TOutput>(
                inner,
                serviceProvider.GetRequiredService<IBatchPartitioner<TInput>>(),
                inputBatch,
                outputBatch,
                serviceProvider.GetService<IPartitionScheduler>()));
    }

    public PipelineStageBuilder<TInput, TOutput> UseOrdering(
        IReadOnlyIndexedBatch<TInput> inputBatch,
        IWritableIndexedBatch<TOutput> outputBatch)
    {
        return Use((serviceProvider, inner) =>
            new OrderingStep<TInput, TOutput>(
                inner,
                serviceProvider.GetRequiredService<IIndexOrdering<TInput>>(),
                inputBatch,
                outputBatch));
    }

    internal IStep<TInput, TOutput> Build(
        IServiceProvider serviceProvider,
        IStep<TInput, TOutput> step)
    {
        IStep<TInput, TOutput> current = step;
        for (int i = _decorators.Count - 1; i >= 0; i--)
        {
            current = _decorators[i](serviceProvider, current);
        }

        return current;
    }
}
