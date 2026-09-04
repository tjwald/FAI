using FAI.Core.Pipelines;

namespace FAI.Core.Extensions.DI;

public sealed class ComposedPipelineBuilder<TStart, TCurrent> : IPipelineBuilder<TStart, TCurrent>
{
    private readonly IServiceCollection _services;
    private readonly Func<IServiceProvider, IPipeline<TStart, TCurrent>> _build;

    internal ComposedPipelineBuilder(IServiceCollection services, Func<IServiceProvider, IPipeline<TStart, TCurrent>> build) => (_services, _build) = (services, build);

    public ComposedPipelineBuilder<TStart, TNext> Then<TNext, TPipeline>() where TPipeline : class, IPipeline<TCurrent, TNext>
    {
        _services.AddSingleton<TPipeline>();
        return Then(serviceProvider => serviceProvider.GetRequiredService<TPipeline>());
    }

    public ComposedPipelineBuilder<TStart, TNext> Then<TNext>(Func<IServiceProvider, IPipeline<TCurrent, TNext>> pipelineFactory)
        => new(_services, serviceProvider => AppendedPipeline.Create(_build(serviceProvider), pipelineFactory(serviceProvider)));

    public ComposedPipelineBuilder<TStart, TNext> Then<TNext>(Func<PipelineBuilder<TCurrent>, IPipelineBuilder<TCurrent, TNext>> buildPipeline)
    {
        IPipelineBuilder<TCurrent, TNext> pipeline = buildPipeline(new PipelineBuilder<TCurrent>(_services));
        return Then(pipeline.BuildPipeline);
    }

    public ComposedPipelineBuilder<TStart, (TCurrent Input, TBranch Output)> Fork<TBranch>(
        Func<PipelineBuilder<TCurrent>, IPipelineBuilder<TCurrent, TBranch>> branch)
    {
        IPipelineBuilder<TCurrent, TBranch> pipeline = branch(new PipelineBuilder<TCurrent>(_services));
        return Then(serviceProvider => new ForkPipeline<TCurrent, TBranch>(pipeline.BuildPipeline(serviceProvider)));
    }

    public ComposedPipelineBuilder<TStart, (T1 Branch1, T2 Branch2)> Fork<T1, T2>(
        Func<PipelineBuilder<TCurrent>, IPipelineBuilder<TCurrent, T1>> branch1,
        Func<PipelineBuilder<TCurrent>, IPipelineBuilder<TCurrent, T2>> branch2)
    {
        IPipelineBuilder<TCurrent, T1> p1 = branch1(new PipelineBuilder<TCurrent>(_services));
        IPipelineBuilder<TCurrent, T2> p2 = branch2(new PipelineBuilder<TCurrent>(_services));
        return Then(serviceProvider => new ForkPipeline<TCurrent, T1, T2>(p1.BuildPipeline(serviceProvider), p2.BuildPipeline(serviceProvider)));
    }

    public DecoratedPipelineBuilder<TStart, TCurrent, TCurrent> Use(IForwardPipelineDecorator<TCurrent> decorator)
        => new(_services, _build, _ => new IdentityPipeline<TCurrent>(), [decorator], true);

    public IServiceCollection Build(string? key = null)
    {
        if (key is null) _services.AddSingleton(_build);
        else _services.AddKeyedSingleton(key, (serviceProvider, _) => _build(serviceProvider));
        return _services;
    }

    public IPipeline<TStart, TCurrent> BuildPipeline(IServiceProvider serviceProvider) => _build(serviceProvider);
}
