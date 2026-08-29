using FAI.Core.Pipelines;

namespace FAI.Core.Extensions.DI;

public delegate bool TryAllocatePipelineOutput<in TInput, TOutput>(TInput input, out TOutput output);

public interface IForwardPipelineDecorator<TInput>
{
    IPipeline<TInput, TOutput> Apply<TOutput>(
        IServiceProvider serviceProvider,
        IPipeline<TInput, TOutput> pipeline);
}

public sealed class PipelineBuilder<TInput>
{
    private readonly IServiceCollection _services;

    internal PipelineBuilder(IServiceCollection services)
    {
        _services = services;
    }

    public DecoratedPipelineBuilder<TInput, TInput, TInput> Use(
        IForwardPipelineDecorator<TInput> decorator)
        => new(
            _services,
            buildPrefix: null,
            serviceProvider => PipelineChain.Create(new IdentityPipeline<TInput>()),
            [decorator],
            suffixIsEmpty: true);

    public ComposedPipelineBuilder<TInput, TOutput> Then<TOutput, TPipeline>()
        where TPipeline : class, IPipeline<TInput, TOutput>
    {
        _services.AddSingleton<TPipeline>();
        return Then(serviceProvider => serviceProvider.GetRequiredService<TPipeline>());
    }

    public ComposedPipelineBuilder<TInput, TOutput> Then<TOutput>(
        Func<IServiceProvider, IPipeline<TInput, TOutput>> pipelineFactory)
        => new(
            _services,
            serviceProvider => PipelineChain.Create(pipelineFactory(serviceProvider)));

    public ComposedPipelineBuilder<TInput, TOutput> Then<TOutput>(
        Func<PipelineBuilder<TInput>, ComposedPipelineBuilder<TInput, TOutput>> buildPipeline)
    {
        ComposedPipelineBuilder<TInput, TOutput> pipeline = buildPipeline(new PipelineBuilder<TInput>(_services));
        return Then(serviceProvider => pipeline.BuildChain(serviceProvider));
    }

    public ComposedPipelineBuilder<TInput, TOutput> Then<TOutput>(
        Func<PipelineBuilder<TInput>, DecoratedPipelineBuilder<TInput, TInput, TOutput>> buildPipeline)
    {
        DecoratedPipelineBuilder<TInput, TInput, TOutput> pipeline = buildPipeline(new PipelineBuilder<TInput>(_services));
        return Then(serviceProvider => pipeline.BuildChain(serviceProvider));
    }

}

public sealed class ComposedPipelineBuilder<TStart, TCurrent>
{
    private readonly IServiceCollection _services;
    private readonly Func<IServiceProvider, IPipelineChain<TStart, TCurrent>> _build;

    internal ComposedPipelineBuilder(
        IServiceCollection services,
        Func<IServiceProvider, IPipelineChain<TStart, TCurrent>> build)
    {
        _services = services;
        _build = build;
    }

    public ComposedPipelineBuilder<TStart, TNext> Then<TNext, TPipeline>()
        where TPipeline : class, IPipeline<TCurrent, TNext>
    {
        _services.AddSingleton<TPipeline>();
        return Then(serviceProvider => serviceProvider.GetRequiredService<TPipeline>());
    }

    public ComposedPipelineBuilder<TStart, TNext> Then<TNext>(
        Func<IServiceProvider, IPipeline<TCurrent, TNext>> pipelineFactory)
        => new(
            _services,
            serviceProvider => new AppendedPipelineChain<TStart, TCurrent, TNext>(
                _build(serviceProvider),
                pipelineFactory(serviceProvider)));

    public ComposedPipelineBuilder<TStart, TNext> Then<TNext>(
        Func<PipelineBuilder<TCurrent>, ComposedPipelineBuilder<TCurrent, TNext>> buildPipeline)
    {
        ComposedPipelineBuilder<TCurrent, TNext> pipeline = buildPipeline(new PipelineBuilder<TCurrent>(_services));
        return Then(serviceProvider => pipeline.BuildChain(serviceProvider));
    }

    public ComposedPipelineBuilder<TStart, TNext> Then<TNext>(
        Func<PipelineBuilder<TCurrent>, DecoratedPipelineBuilder<TCurrent, TCurrent, TNext>> buildPipeline)
    {
        DecoratedPipelineBuilder<TCurrent, TCurrent, TNext> pipeline = buildPipeline(new PipelineBuilder<TCurrent>(_services));
        return Then(serviceProvider => pipeline.BuildChain(serviceProvider));
    }

    public ComposedPipelineBuilder<TStart, TCurrent> WithOutputAllocation(
        TryAllocatePipelineOutput<TStart, TCurrent> tryAllocateOutput)
    {
        return new ComposedPipelineBuilder<TStart, TCurrent>(
            _services,
            serviceProvider => PipelineChain.Create(
                new PreallocatingPipeline<TStart, TCurrent>(
                    _build(serviceProvider),
                    tryAllocateOutput)));
    }

    public DecoratedPipelineBuilder<TStart, TCurrent, TCurrent> Use(
        IForwardPipelineDecorator<TCurrent> decorator)
        => new(
            _services,
            _build,
            serviceProvider => PipelineChain.Create(new IdentityPipeline<TCurrent>()),
            [decorator],
            suffixIsEmpty: true);

    public IServiceCollection Build(string? key = null)
    {
        if (key is null)
        {
            _services.AddSingleton<IPipeline<TStart, TCurrent>>(serviceProvider => _build(serviceProvider));
        }
        else
        {
            _services.AddKeyedSingleton<IPipeline<TStart, TCurrent>>(key, (serviceProvider, _) => _build(serviceProvider));
        }

        return _services;
    }

    internal IPipelineChain<TStart, TCurrent> BuildChain(IServiceProvider serviceProvider) => _build(serviceProvider);
}

public sealed class DecoratedPipelineBuilder<TStart, TBoundary, TCurrent>
{
    private readonly IServiceCollection _services;
    private readonly Func<IServiceProvider, IPipelineChain<TStart, TBoundary>>? _buildPrefix;
    private readonly Func<IServiceProvider, IPipelineChain<TBoundary, TCurrent>> _buildSuffix;
    private readonly IReadOnlyList<IForwardPipelineDecorator<TBoundary>> _decorators;
    private readonly bool _suffixIsEmpty;

    internal DecoratedPipelineBuilder(
        IServiceCollection services,
        Func<IServiceProvider, IPipelineChain<TStart, TBoundary>>? buildPrefix,
        Func<IServiceProvider, IPipelineChain<TBoundary, TCurrent>> buildSuffix,
        IReadOnlyList<IForwardPipelineDecorator<TBoundary>> decorators,
        bool suffixIsEmpty)
    {
        _services = services;
        _buildPrefix = buildPrefix;
        _buildSuffix = buildSuffix;
        _decorators = decorators;
        _suffixIsEmpty = suffixIsEmpty;
    }

    public DecoratedPipelineBuilder<TStart, TBoundary, TNext> Then<TNext, TPipeline>()
        where TPipeline : class, IPipeline<TCurrent, TNext>
    {
        _services.AddSingleton<TPipeline>();
        return Then(serviceProvider => serviceProvider.GetRequiredService<TPipeline>());
    }

    public DecoratedPipelineBuilder<TStart, TBoundary, TNext> Then<TNext>(
        Func<IServiceProvider, IPipeline<TCurrent, TNext>> pipelineFactory)
        => new(
            _services,
            _buildPrefix,
            serviceProvider => Append(serviceProvider, pipelineFactory),
            _decorators,
            suffixIsEmpty: false);

    public DecoratedPipelineBuilder<TStart, TBoundary, TNext> Then<TNext>(
        Func<PipelineBuilder<TCurrent>, ComposedPipelineBuilder<TCurrent, TNext>> buildPipeline)
    {
        ComposedPipelineBuilder<TCurrent, TNext> pipeline = buildPipeline(new PipelineBuilder<TCurrent>(_services));
        return Then(serviceProvider => pipeline.BuildChain(serviceProvider));
    }

    public DecoratedPipelineBuilder<TStart, TBoundary, TNext> Then<TNext>(
        Func<PipelineBuilder<TCurrent>, DecoratedPipelineBuilder<TCurrent, TCurrent, TNext>> buildPipeline)
    {
        DecoratedPipelineBuilder<TCurrent, TCurrent, TNext> pipeline = buildPipeline(new PipelineBuilder<TCurrent>(_services));
        return Then(serviceProvider => pipeline.BuildChain(serviceProvider));
    }

    public DecoratedPipelineBuilder<TStart, TBoundary, TCurrent> Use(
        IForwardPipelineDecorator<TBoundary> decorator)
        => new(
            _services,
            _buildPrefix,
            _buildSuffix,
            [.. _decorators, decorator],
            _suffixIsEmpty);

    public DecoratedPipelineBuilder<TStart, TBoundary, TCurrent> WithOutputAllocation(
        TryAllocatePipelineOutput<TBoundary, TCurrent> tryAllocateOutput)
        => new(
            _services,
            _buildPrefix,
            serviceProvider => PipelineChain.Create(
                new PreallocatingPipeline<TBoundary, TCurrent>(
                    _buildSuffix(serviceProvider),
                    tryAllocateOutput)),
                    _decorators,
                    suffixIsEmpty: false);

    public IServiceCollection Build(string? key = null)
    {
        if (key is null)
        {
            _services.AddSingleton<IPipeline<TStart, TCurrent>>(BuildPipeline);
        }
        else
        {
            _services.AddKeyedSingleton<IPipeline<TStart, TCurrent>>(key, (serviceProvider, _) => BuildPipeline(serviceProvider));
        }

        return _services;
    }

    internal IPipelineChain<TStart, TCurrent> BuildChain(IServiceProvider serviceProvider)
    {
        IPipeline<TBoundary, TCurrent> suffix = BuildDecoratedSuffix(serviceProvider);
        if (_buildPrefix is null)
        {
            return (IPipelineChain<TStart, TCurrent>)(object)PipelineChain.Create(suffix);
        }

        return new AppendedPipelineChain<TStart, TBoundary, TCurrent>(
            _buildPrefix(serviceProvider),
            suffix);
    }

    private IPipelineChain<TStart, TCurrent> BuildPipeline(IServiceProvider serviceProvider)
        => BuildChain(serviceProvider);

    private IPipeline<TBoundary, TCurrent> BuildDecoratedSuffix(IServiceProvider serviceProvider)
    {
        IPipeline<TBoundary, TCurrent> suffix = _buildSuffix(serviceProvider);
        for (int index = _decorators.Count - 1; index >= 0; index--)
        {
            suffix = _decorators[index].Apply(serviceProvider, suffix);
        }

        return suffix;
    }

    private IPipelineChain<TBoundary, TNext> Append<TNext>(
        IServiceProvider serviceProvider,
        Func<IServiceProvider, IPipeline<TCurrent, TNext>> pipelineFactory)
    {
        IPipeline<TCurrent, TNext> pipeline = pipelineFactory(serviceProvider);
        if (_suffixIsEmpty)
        {
            return (IPipelineChain<TBoundary, TNext>)(object)PipelineChain.Create(pipeline);
        }

        return new AppendedPipelineChain<TBoundary, TCurrent, TNext>(
            _buildSuffix(serviceProvider),
            pipeline);
    }
}

internal sealed class IdentityPipeline<T> : IPipeline<T, T>
{
    public ValueTask<T> ExecuteAsync(T input, CancellationToken cancellationToken = default)
        => ValueTask.FromResult(input);
}

internal sealed class PreallocatingPipeline<TInput, TOutput> : IPreallocatingPipeline<TInput, TOutput>
{
    private readonly IPipelineChain<TInput, TOutput> _pipeline;
    private readonly TryAllocatePipelineOutput<TInput, TOutput> _tryAllocateOutput;

    public PreallocatingPipeline(
        IPipelineChain<TInput, TOutput> pipeline,
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

