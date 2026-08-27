namespace FAI.Core.Extensions.DI;

public static class FaiBuilderExtensions
{
    extension(IServiceCollection services)
    {
        public PipelineBuilder<TInput> AddPipeline<TInput>()
        {
            return new PipelineBuilder<TInput>(services);
        }
    }
}
