using FAI.Core.Pipelines;

namespace FAI.Core.Extensions.DI;

public delegate bool TryAllocatePipelineOutput<in TInput, TOutput>(TInput input, out TOutput output);

public interface IForwardPipelineDecorator<TInput>
{
    IPipeline<TInput, TOutput> Apply<TOutput>(IServiceProvider serviceProvider, IPipeline<TInput, TOutput> pipeline);
}
