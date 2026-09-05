namespace FAI.Core.Extensions.DI;

public static class FaiBuilderExtensions
{
    extension(IServiceCollection services)
    {
        public PipelineBuilder<TInput, TInput> AddPipeline<TInput>()
        {
            services.AddIndexedBatchRegistry();
            return new PipelineBuilder<TInput, TInput>(services, new InitialStage<TInput>());
        }
    }
}
