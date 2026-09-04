using FAI.Core.Pipelines;

namespace FAI.Core.Extensions.DI;

public interface IForwardPipelineDecorator<TInput>
{
    IPipeline<TInput, TOutput> Apply<TOutput>(IServiceProvider serviceProvider, IPipeline<TInput, TOutput> pipeline);
}
