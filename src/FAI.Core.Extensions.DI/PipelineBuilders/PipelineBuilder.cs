using FAI.Core.Pipelines;

namespace FAI.Core.Extensions.DI;

public sealed class PipelineBuilder<TInput>
{
    private readonly IServiceCollection _services;

    internal PipelineBuilder(IServiceCollection services) => _services = services;

    public DecoratedPipelineBuilder<TInput, TInput, TInput> Use(IForwardPipelineDecorator<TInput> decorator)
        => new(_services, null, _ => new IdentityPipeline<TInput>(), [decorator], true);

    public ComposedPipelineBuilder<TInput, TOutput> Then<TOutput, TPipeline>() where TPipeline : class, IPipeline<TInput, TOutput>
    {
        _services.AddSingleton<TPipeline>();
        return Then(serviceProvider => serviceProvider.GetRequiredService<TPipeline>());
    }

    public ComposedPipelineBuilder<TInput, TOutput> Then<TOutput>(Func<IServiceProvider, IPipeline<TInput, TOutput>> pipelineFactory)
        => new(_services, pipelineFactory);

    public ComposedPipelineBuilder<TInput, TOutput> Then<TOutput>(Func<PipelineBuilder<TInput>, IPipelineBuilder<TInput, TOutput>> buildPipeline)
    {
        IPipelineBuilder<TInput, TOutput> pipeline = buildPipeline(new PipelineBuilder<TInput>(_services));
        return Then(pipeline.BuildPipeline);
    }

    public ComposedPipelineBuilder<TInput, (TInput Input, TBranch Output)> Fork<TBranch>(
        Func<PipelineBuilder<TInput>, IPipelineBuilder<TInput, TBranch>> branch)
    {
        IPipelineBuilder<TInput, TBranch> pipeline = branch(new PipelineBuilder<TInput>(_services));
        return Then(serviceProvider => new ForkPipeline<TInput, TBranch>(pipeline.BuildPipeline(serviceProvider)));
    }

    public ComposedPipelineBuilder<TInput, (T1 Branch1, T2 Branch2)> Fork<T1, T2>(
        Func<PipelineBuilder<TInput>, IPipelineBuilder<TInput, T1>> branch1,
        Func<PipelineBuilder<TInput>, IPipelineBuilder<TInput, T2>> branch2)
    {
        IPipelineBuilder<TInput, T1> p1 = branch1(new PipelineBuilder<TInput>(_services));
        IPipelineBuilder<TInput, T2> p2 = branch2(new PipelineBuilder<TInput>(_services));
        return Then(serviceProvider => new ForkPipeline<TInput, T1, T2>(p1.BuildPipeline(serviceProvider), p2.BuildPipeline(serviceProvider)));
    }
}
