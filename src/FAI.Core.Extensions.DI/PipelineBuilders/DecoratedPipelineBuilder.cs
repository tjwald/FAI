using FAI.Core.Pipelines;

namespace FAI.Core.Extensions.DI;

public sealed class DecoratedPipelineBuilder<TStart, TBoundary, TCurrent>
{
    private readonly IServiceCollection _services;
    private readonly Func<IServiceProvider, IPipelineChain<TStart, TBoundary>>? _buildPrefix;
    private readonly Func<IServiceProvider, IPipelineChain<TBoundary, TCurrent>> _buildSuffix;
    private readonly IReadOnlyList<IForwardPipelineDecorator<TBoundary>> _decorators;
    private readonly bool _suffixIsEmpty;

    internal DecoratedPipelineBuilder(IServiceCollection services, Func<IServiceProvider, IPipelineChain<TStart, TBoundary>>? buildPrefix, Func<IServiceProvider, IPipelineChain<TBoundary, TCurrent>> buildSuffix, IReadOnlyList<IForwardPipelineDecorator<TBoundary>> decorators, bool suffixIsEmpty)
        => (_services, _buildPrefix, _buildSuffix, _decorators, _suffixIsEmpty) = (services, buildPrefix, buildSuffix, decorators, suffixIsEmpty);

    public DecoratedPipelineBuilder<TStart, TBoundary, TNext> Then<TNext, TPipeline>() where TPipeline : class, IPipeline<TCurrent, TNext>
    {
        _services.AddSingleton<TPipeline>();
        return Then(serviceProvider => serviceProvider.GetRequiredService<TPipeline>());
    }

    public DecoratedPipelineBuilder<TStart, TBoundary, TNext> Then<TNext>(Func<IServiceProvider, IPipeline<TCurrent, TNext>> pipelineFactory)
        => new(_services, _buildPrefix, serviceProvider => Append(serviceProvider, pipelineFactory), _decorators, false);

    public DecoratedPipelineBuilder<TStart, TBoundary, TNext> Then<TNext>(Func<PipelineBuilder<TCurrent>, ComposedPipelineBuilder<TCurrent, TNext>> buildPipeline)
    {
        ComposedPipelineBuilder<TCurrent, TNext> pipeline = buildPipeline(new PipelineBuilder<TCurrent>(_services));
        return Then(pipeline.BuildChain);
    }

    public DecoratedPipelineBuilder<TStart, TBoundary, TNext> Then<TNext>(Func<PipelineBuilder<TCurrent>, DecoratedPipelineBuilder<TCurrent, TCurrent, TNext>> buildPipeline)
    {
        DecoratedPipelineBuilder<TCurrent, TCurrent, TNext> pipeline = buildPipeline(new PipelineBuilder<TCurrent>(_services));
        return Then(pipeline.BuildChain);
    }

    public DecoratedPipelineBuilder<TStart, TBoundary, TCurrent> Use(IForwardPipelineDecorator<TBoundary> decorator)
        => new(_services, _buildPrefix, _buildSuffix, [.. _decorators, decorator], _suffixIsEmpty);

    public IServiceCollection Build(string? key = null)
    {
        if (key is null) _services.AddSingleton<IPipeline<TStart, TCurrent>>(BuildPipeline);
        else _services.AddKeyedSingleton<IPipeline<TStart, TCurrent>>(key, (serviceProvider, _) => BuildPipeline(serviceProvider));
        return _services;
    }

    internal IPipelineChain<TStart, TCurrent> BuildChain(IServiceProvider serviceProvider)
    {
        IPipeline<TBoundary, TCurrent> suffix = BuildDecoratedSuffix(serviceProvider);
        return _buildPrefix is null
            ? (IPipelineChain<TStart, TCurrent>)(object)PipelineChain.Create(suffix)
            : new AppendedPipelineChain<TStart, TBoundary, TCurrent>(_buildPrefix(serviceProvider), suffix);
    }

    private IPipelineChain<TStart, TCurrent> BuildPipeline(IServiceProvider serviceProvider) => BuildChain(serviceProvider);

    private IPipeline<TBoundary, TCurrent> BuildDecoratedSuffix(IServiceProvider serviceProvider)
    {
        IPipeline<TBoundary, TCurrent> suffix = _buildSuffix(serviceProvider);
        for (int index = _decorators.Count - 1; index >= 0; index--) suffix = _decorators[index].Apply(serviceProvider, suffix);
        return suffix;
    }

    private IPipelineChain<TBoundary, TNext> Append<TNext>(IServiceProvider serviceProvider, Func<IServiceProvider, IPipeline<TCurrent, TNext>> pipelineFactory)
    {
        IPipeline<TCurrent, TNext> pipeline = pipelineFactory(serviceProvider);
        return _suffixIsEmpty
            ? (IPipelineChain<TBoundary, TNext>)(object)PipelineChain.Create(pipeline)
            : new AppendedPipelineChain<TBoundary, TCurrent, TNext>(_buildSuffix(serviceProvider), pipeline);
    }
}
