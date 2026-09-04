using FAI.Core.Pipelines;

namespace FAI.Core.Extensions.DI;

public sealed class DecoratedPipelineBuilder<TStart, TBoundary, TCurrent>
{
    private readonly IServiceCollection _services;
    private readonly Func<IServiceProvider, IPipeline<TStart, TBoundary>>? _buildPrefix;
    private readonly Func<IServiceProvider, IPipeline<TBoundary, TCurrent>> _buildSuffix;
    private readonly IReadOnlyList<IForwardPipelineDecorator<TBoundary>> _decorators;
    private readonly bool _suffixIsEmpty;

    internal DecoratedPipelineBuilder(IServiceCollection services, Func<IServiceProvider, IPipeline<TStart, TBoundary>>? buildPrefix, Func<IServiceProvider, IPipeline<TBoundary, TCurrent>> buildSuffix, IReadOnlyList<IForwardPipelineDecorator<TBoundary>> decorators, bool suffixIsEmpty)
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
        return Then(pipeline.BuildPipeline);
    }

    public DecoratedPipelineBuilder<TStart, TBoundary, TNext> Then<TNext>(Func<PipelineBuilder<TCurrent>, DecoratedPipelineBuilder<TCurrent, TCurrent, TNext>> buildPipeline)
    {
        DecoratedPipelineBuilder<TCurrent, TCurrent, TNext> pipeline = buildPipeline(new PipelineBuilder<TCurrent>(_services));
        return Then(pipeline.BuildPipeline);
    }

    public DecoratedPipelineBuilder<TStart, TBoundary, TCurrent> Use(IForwardPipelineDecorator<TBoundary> decorator)
        => new(_services, _buildPrefix, _buildSuffix, [.. _decorators, decorator], _suffixIsEmpty);

    public IServiceCollection Build(string? key = null)
    {
        if (key is null) _services.AddSingleton(BuildPipeline);
        else _services.AddKeyedSingleton(key, (serviceProvider, _) => BuildPipeline(serviceProvider));
        return _services;
    }

    internal IPipeline<TStart, TCurrent> BuildPipeline(IServiceProvider serviceProvider)
    {
        IPipeline<TBoundary, TCurrent> suffix = BuildDecoratedSuffix(serviceProvider);
        return _buildPrefix is null
            ? (IPipeline<TStart, TCurrent>)(object)suffix
            : AppendedPipeline.Create(_buildPrefix(serviceProvider), suffix);
    }

    private IPipeline<TBoundary, TCurrent> BuildDecoratedSuffix(IServiceProvider serviceProvider)
    {
        IPipeline<TBoundary, TCurrent> suffix = _buildSuffix(serviceProvider);
        for (int index = _decorators.Count - 1; index >= 0; index--) suffix = _decorators[index].Apply(serviceProvider, suffix);
        return suffix;
    }

    private IPipeline<TBoundary, TNext> Append<TNext>(IServiceProvider serviceProvider, Func<IServiceProvider, IPipeline<TCurrent, TNext>> pipelineFactory)
    {
        IPipeline<TCurrent, TNext> pipeline = pipelineFactory(serviceProvider);
        return _suffixIsEmpty
            ? (IPipeline<TBoundary, TNext>)(object)pipeline
            : AppendedPipeline.Create(_buildSuffix(serviceProvider), pipeline);
    }
}
