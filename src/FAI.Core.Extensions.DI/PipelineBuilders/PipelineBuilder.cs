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
}
