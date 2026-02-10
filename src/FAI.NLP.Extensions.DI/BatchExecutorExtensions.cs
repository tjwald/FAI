using FAI.Core.Abstractions;
using FAI.Core.Configurations.InferenceTasks;
using FAI.Core.Extensions.DI;
using FAI.Core.ResultTypes;
using FAI.NLP.BatchSlicer;
using FAI.NLP.Configuration.PipelineBatchExecutors;
using FAI.NLP.InferenceTasks.TextClassification;
using FAI.NLP.PipelineBatchExecutors;
using FAI.NLP.Tokenization;
using Microsoft.Extensions.DependencyInjection;

namespace FAI.NLP.Extensions.DI;

public static class BatchExecutorExtensions
{
    extension<TInput, TOutput>(PipelineBuilder<TInput, TOutput> builder) where TInput : ITokenizable
    {
        public PipelineBuilder<TInput, TOutput> UseTokenSorting(TokenCountSortingBatchExecutorOptions? options = null)
        {
            options ??= new();
            return builder.Use<TokenCountSortingBatchExecutor<TInput, TOutput>>((next, sp)
                => ActivatorUtilities.CreateInstance<TokenCountSortingBatchExecutor<TInput, TOutput>>(sp, next, options));
        }

        public PipelineBuilder<TInput, TOutput> UseTokenSorting(string section)
        {
            builder.AddServices(sp => sp.AddConfigurationAndBind<TokenCountSortingBatchExecutorOptions>(section));
            return builder.Use<TokenCountSortingBatchExecutor<TInput, TOutput>>();
        }
    }

    extension<TInput, TOutput>(PipelineBuilder<TInput, TOutput> builder) where TInput : ITokenizable
    {
        public PipelineBuilder<TInput, TOutput> UseTokenizing()
        {
            return builder.Use<TokenizerBatchExecutor<TInput, TOutput>>();
        }
    }

    extension<TClassification>(PipelineBuilder<TokenizedText, ClassificationResult<TClassification, float>> builder)
    {
        public PipelineBuilder<TokenizedText, ClassificationResult<TClassification, float>> WithTextClassification(string section)
        {
            builder.AddServices(serviceCollection => serviceCollection.AddConfigurationAndBind<ClassificationOptions<TClassification>>(section));
            builder.AddInferenceSteps<TextClassification<TClassification>>();
            return builder;
        }
    }

    extension<TInput, TOutput>(PartitionBatchExecutorBuilder<TInput, TOutput> builder) where TInput : ITokenizable
    {
        public PartitionBatchExecutorBuilder<TInput, TOutput> WithMaxPaddedTokens(string section)
        {
            builder.AddServices(serviceCollection =>
            {
                serviceCollection.AddConfigurationAndBind<MaxPaddedTokensSlicerOptions>(section);
                serviceCollection.AddSingleton<IBatchSlicer<TInput>, MaxPaddedTokensBatchSlicer<TInput>>();
            });
            return builder;
        }
    }
}
