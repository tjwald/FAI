using FAI.Core.Pipelines;

namespace FAI.Core.Extensions.DI;

public sealed class ComposedPipelineBuilder<TStart, TCurrent>
{
    private readonly IServiceCollection _services;
    private readonly Func<IServiceProvider, IPipelineChain<TStart, TCurrent>> _build;

    internal ComposedPipelineBuilder(IServiceCollection services, Func<IServiceProvider, IPipelineChain<TStart, TCurrent>> build) => (_services, _build) = (services, build);

    public ComposedPipelineBuilder<TStart, TNext> Then<TNext, TPipeline>() where TPipeline : class, IPipeline<TCurrent, TNext>
    {
        _services.AddSingleton<TPipeline>();
        return Then(serviceProvider => serviceProvider.GetRequiredService<TPipeline>());
    }

    public ComposedPipelineBuilder<TStart, TNext> Then<TNext>(Func<IServiceProvider, IPipeline<TCurrent, TNext>> pipelineFactory)
        => new(_services, serviceProvider => new AppendedPipelineChain<TStart, TCurrent, TNext>(_build(serviceProvider), pipelineFactory(serviceProvider)));

    public ComposedPipelineBuilder<TStart, TNext> Then<TNext>(Func<PipelineBuilder<TCurrent>, ComposedPipelineBuilder<TCurrent, TNext>> buildPipeline)
    {
        ComposedPipelineBuilder<TCurrent, TNext> pipeline = buildPipeline(new PipelineBuilder<TCurrent>(_services));
        return Then(pipeline.BuildChain);
    }

    public ComposedPipelineBuilder<TStart, TNext> Then<TNext>(Func<PipelineBuilder<TCurrent>, DecoratedPipelineBuilder<TCurrent, TCurrent, TNext>> buildPipeline)
    {
        DecoratedPipelineBuilder<TCurrent, TCurrent, TNext> pipeline = buildPipeline(new PipelineBuilder<TCurrent>(_services));
        return Then(pipeline.BuildChain);
    }

    public DecoratedPipelineBuilder<TStart, TCurrent, TCurrent> Use(IForwardPipelineDecorator<TCurrent> decorator)
        => new(_services, _build, _ => PipelineChain.Create(new IdentityPipeline<TCurrent>()), [decorator], true);

    public IServiceCollection Build(string? key = null)
    {
        if (key is null) _services.AddSingleton<IPipeline<TStart, TCurrent>>(_build);
        else _services.AddKeyedSingleton<IPipeline<TStart, TCurrent>>(key, (serviceProvider, _) => _build(serviceProvider));
        return _services;
    }

    internal IPipelineChain<TStart, TCurrent> BuildChain(IServiceProvider serviceProvider) => _build(serviceProvider);
}
