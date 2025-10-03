namespace FAI.Core.Extensions.DI;

public static class FaiBuilderExtensions
{
    public static PipelineBuilder<TInput, TOutput> AddPipelineBuilder<TInput, TOutput>(this IServiceCollection services)
    {
        return new PipelineBuilder<TInput, TOutput>(services);
    }

    public static DecoratorChainBuilder AddDecoratedChain(this IServiceCollection services)
    {
        return new DecoratorChainBuilder(services);
    }
}
