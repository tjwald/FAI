using FAI.Core.Extensions.DI;
using FAI.Core.Steps;
using FAI.NLP.Configuration;
using FAI.NLP.Steps;
using FAI.NLP.Tokenization;
using Microsoft.Extensions.DependencyInjection;

namespace FAI.NLP.Extensions.DI;

public static class BatchExecutorExtensions
{
    extension<TInput, TOutput>(PipelineStageBuilder<ReadOnlyMemory<TInput>, Memory<TOutput>> stage)
        where TInput : ITokenizable
    {
        public PipelineStageBuilder<ReadOnlyMemory<TInput>, Memory<TOutput>> UseTokenizingStep()
        {
            return stage.Use((serviceProvider, inner) =>
                new TokenizingStep<TInput, TOutput>(
                    inner,
                    serviceProvider.GetRequiredService<PretrainedTokenizer>()));
        }

        public PipelineStageBuilder<ReadOnlyMemory<TInput>, Memory<TOutput>> UseTokenCountOrderingStep()
        {
            return stage.Use((serviceProvider, inner) =>
                new OrderingStep<
                    ReadOnlyMemory<TInput>,
                    Memory<TOutput>,
                    ReadOnlyMemoryBatchOperations<TInput>,
                    MemoryBatchOperations<TOutput>>(
                        inner,
                        new TokenCountOrdering<TInput>(
                            serviceProvider.GetRequiredService<TokenCountOrderingOptions>())));
        }

        public PipelineStageBuilder<ReadOnlyMemory<TInput>, Memory<TOutput>> UseMaxPaddedTokensPartitioningStep()
        {
            return stage.Use((serviceProvider, inner) =>
                new PartitioningStep<
                    ReadOnlyMemory<TInput>,
                    Memory<TOutput>,
                    ReadOnlyMemoryBatchOperations<TInput>,
                    MemoryBatchOperations<TOutput>>(
                        inner,
                        new MaxPaddedTokensPartitioner<TInput>(
                            serviceProvider.GetRequiredService<MaxPaddedTokensPartitionerOptions>()),
                        serviceProvider.GetService<IPartitionScheduler>()));
        }
    }

}
