using FAI.Core.Pipelines;

namespace FAI.Core.Extensions.DI;

public sealed class PipelineBuilder<TStart, TCurrent> : IPipelineBuilder<TStart, TCurrent>
{
    private readonly IServiceCollection _services;
    private readonly IStage<TStart, TCurrent> _stage;

    internal PipelineBuilder(IServiceCollection services, IStage<TStart, TCurrent> stage)
    {
        _services = services;
        _stage = stage;
    }

    public PipelineBuilder<TStart, TNext> Then<TNext, TPipeline>() where TPipeline : class, IPipeline<TCurrent, TNext>
    {
        _services.AddSingleton<TPipeline>();
        return Then(serviceProvider => serviceProvider.GetRequiredService<TPipeline>());
    }

    public PipelineBuilder<TStart, TNext> Then<TNext>(Func<IServiceProvider, IPipeline<TCurrent, TNext>> pipelineFactory)
        => new(_services, _stage.Append(pipelineFactory));

    public PipelineBuilder<TStart, TNext> Then<TNext>(Func<PipelineBuilder<TCurrent, TCurrent>, IPipelineBuilder<TCurrent, TNext>> buildPipeline)
    {
        IPipelineBuilder<TCurrent, TNext> pipeline = buildPipeline(new PipelineBuilder<TCurrent, TCurrent>(_services, new InitialStage<TCurrent>()));
        return Then(pipeline.BuildPipeline);
    }

    public PipelineBuilder<TStart, (TCurrent Input, TBranch Output)> Fork<TBranch>(
        Func<PipelineBuilder<TCurrent, TCurrent>, IPipelineBuilder<TCurrent, TBranch>> branch)
    {
        IPipelineBuilder<TCurrent, TBranch> pipeline = branch(new PipelineBuilder<TCurrent, TCurrent>(_services, new InitialStage<TCurrent>()));
        return Then(serviceProvider => new ForkPipeline<TCurrent, TBranch>(pipeline.BuildPipeline(serviceProvider)));
    }

    public PipelineBuilder<TStart, (T1 Branch1, T2 Branch2)> Fork<T1, T2>(
        Func<PipelineBuilder<TCurrent, TCurrent>, IPipelineBuilder<TCurrent, T1>> branch1,
        Func<PipelineBuilder<TCurrent, TCurrent>, IPipelineBuilder<TCurrent, T2>> branch2)
    {
        IPipelineBuilder<TCurrent, T1> p1 = branch1(new PipelineBuilder<TCurrent, TCurrent>(_services, new InitialStage<TCurrent>()));
        IPipelineBuilder<TCurrent, T2> p2 = branch2(new PipelineBuilder<TCurrent, TCurrent>(_services, new InitialStage<TCurrent>()));
        return Then(serviceProvider => new ForkPipeline<TCurrent, T1, T2>(p1.BuildPipeline(serviceProvider), p2.BuildPipeline(serviceProvider)));
    }

    public PipelineBuilder<TStart, TCurrent> Use(IForwardPipelineDecorator<TCurrent> decorator)
        => new(_services, _stage.Use(decorator));

    public IServiceCollection Build(string? key = null)
    {
        if (key is null)
        {
            _services.AddSingleton(_stage.Build);
        }
        else
        {
            _services.AddKeyedSingleton(key, (serviceProvider, _) => _stage.Build(serviceProvider));
        }

        return _services;
    }

    public IPipeline<TStart, TCurrent> BuildPipeline(IServiceProvider serviceProvider) => _stage.Build(serviceProvider);
}
