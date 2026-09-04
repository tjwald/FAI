using FAI.Core.Extensions.DI;
using FAI.Core.Pipelines;
using FAI.NLP.Configuration;
using FAI.NLP.Pipelines;
using FAI.NLP.Tokenization;
using Microsoft.Extensions.DependencyInjection;

namespace FAI.NLP.Extensions.DI;

public static class BatchExecutorExtensions
{
    extension<TStart, TInput>(PipelineBuilder<TStart, ReadOnlyMemory<TInput>> pipeline)
        where TInput : ITokenizable
    {
        public PipelineBuilder<TStart, ReadOnlyMemory<TInput>> UseTokenCountOrdering()
            => pipeline.Use(new TokenCountOrderingDecorator<TInput>());

        public PipelineBuilder<TStart, ReadOnlyMemory<TInput>> UseMaxPaddedTokensPartitioning()
            => pipeline.Use(new MaxPaddedTokensPartitioningDecorator<TInput>());
    }

    private sealed class TokenCountOrderingDecorator<TInput> : IForwardPipelineDecorator<ReadOnlyMemory<TInput>>
        where TInput : ITokenizable
    {
        public IPipeline<ReadOnlyMemory<TInput>, TOutput> Apply<TOutput>(
            IServiceProvider serviceProvider,
            IPipeline<ReadOnlyMemory<TInput>, TOutput> pipeline)
            => new OrderingPipeline<ReadOnlyMemory<TInput>, TOutput>(
                pipeline,
                new TokenCountOrdering<TInput>(serviceProvider.GetRequiredService<TokenCountOrderingOptions>()),
                new ReadOnlyMemoryBatchOperations<TInput>(),
                IndexedBatchOperations.GetWritable<TOutput>());
    }

    private sealed class MaxPaddedTokensPartitioningDecorator<TInput> : IForwardPipelineDecorator<ReadOnlyMemory<TInput>>
        where TInput : ITokenizable
    {
        public IPipeline<ReadOnlyMemory<TInput>, TOutput> Apply<TOutput>(
            IServiceProvider serviceProvider,
            IPipeline<ReadOnlyMemory<TInput>, TOutput> pipeline)
            => new PartitioningPipeline<ReadOnlyMemory<TInput>, TOutput>(
                pipeline,
                new MaxPaddedTokensPartitioner<TInput>(
                    serviceProvider.GetRequiredService<MaxPaddedTokensPartitionerOptions>()),
                new ReadOnlyMemoryBatchOperations<TInput>(),
                IndexedBatchOperations.GetWritable<TOutput>(),
                serviceProvider.GetService<IPartitionScheduler>());
    }
}
