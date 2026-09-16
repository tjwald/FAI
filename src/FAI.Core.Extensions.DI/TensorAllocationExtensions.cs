using System.Numerics.Tensors;
using FAI.Core.Pipelines;

namespace FAI.Core.Extensions.DI;

public static class TensorAllocationExtensions
{
    public static PipelineBuilder<T, T> UseTensorDestinationAllocation<T, TElement>(
        this PipelineBuilder<T, T> pipeline,
        params int[] trailingDimensions)
        => UseTensorDestinationAllocation<T, T, TElement>(pipeline, trailingDimensions);

    public static PipelineBuilder<T, T> UseTensorDestinationAllocation<T, TElement>(
        this PipelineBuilder<T, T> pipeline,
        ReadOnlySpan<nint> trailingDimensions = default)
        => UseTensorDestinationAllocation<T, T, TElement>(pipeline, trailingDimensions);

    public static PipelineBuilder<TStart, TCurrent> UseTensorDestinationAllocation<TStart, TCurrent, TElement>(
        this PipelineBuilder<TStart, TCurrent> pipeline,
        params int[] trailingDimensions)
    {
        return pipeline.Use(new TensorDestinationAllocationDecorator<TCurrent, TElement>(
            Array.ConvertAll(trailingDimensions, x => (nint)x)));
    }

    public static PipelineBuilder<TStart, TCurrent> UseTensorDestinationAllocation<TStart, TCurrent, TElement>(
        this PipelineBuilder<TStart, TCurrent> pipeline,
        ReadOnlySpan<nint> trailingDimensions = default)
    {
        return pipeline.Use(new TensorDestinationAllocationDecorator<TCurrent, TElement>(trailingDimensions.ToArray()));
    }

    private sealed class TensorDestinationAllocationDecorator<TInput, TElement>(nint[]? trailingDimensions)
        : IForwardPipelineDecorator<TInput>
    {
        public IPipeline<TInput, TOutput> Apply<TOutput>(
            IServiceProvider serviceProvider,
            IPipeline<TInput, TOutput> pipeline)
        {
            if (pipeline is IPipeline<TInput, Tensor<TElement>> tensorPipeline)
            {
                var inputBatch = serviceProvider.GetService<IReadOnlyIndexedBatch<TInput>>()
                    ?? serviceProvider.GetRequiredReadOnlyBatch<TInput>();

                var allocated = new TensorDestinationAllocationPipeline<TInput, TElement>(
                    tensorPipeline,
                    inputBatch,
                    serviceProvider.GetRequiredWritableBatch<Tensor<TElement>>(),
                    trailingDimensions ?? []);
                return (IPipeline<TInput, TOutput>)(object)allocated;
            }

            return pipeline;
        }
    }
}
