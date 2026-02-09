using FAI.Core.Abstractions;
using FAI.Core.Configurations.InferenceTasks;
using FAI.Core.Configurations.ModelExecutors;
using FAI.Core.Configurations.PipelineBatchExecutors;
using FAI.Core.Extensions.DI;
using FAI.Core.PipelineBatchExecutors;
using FAI.Core.ResultTypes;
using FAI.Extensions.DependencyInjection.LocalServices;
using FAI.NLP.Configuration.PipelineBatchExecutors;
using FAI.NLP.InferenceTasks.TextClassification;
using FAI.NLP.PipelineBatchExecutors;
using FAI.NLP.Tokenization;
using FAI.Onnx.Configuration;
using FAI.Onnx.Factories;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;

namespace Example.SentimentInference.Model;

public static class SentimentInferenceFactory
{
    public static IServiceCollection AddDefaultSentimentInference(this IServiceCollection services, SentimentInferenceOptions options)
    {
        return services.AddLocalServices(localServices =>
        {
            localServices
                .AddConfigurationAndBind<ClassificationOptions<bool>>("SentimentInference:Classification")
                .AddConfigurationAndBind<SerialPipelineBatchExecutorOptions>("SentimentInference:BatchExecutors:SerialPipeline")
                .AddConfigurationAndBind<MaxPaddedTokensBatchExecutorOptions>("SentimentInference:BatchExecutors:MaxPaddedTokens")
                .AddConfigurationAndBind<TokenCountSortingBatchExecutorOptions>("SentimentInference:BatchExecutors:TokenCountSorting");

            localServices.AddSingleton<IModelExecutorOptions, OnnxModelExecutorOptions>(sp => new OnnxModelExecutorOptions()
                .ConfigureOnnxOptions(onnxOptions =>
                {
                    onnxOptions.ConfigureSessionOptions(sessionOptions =>
                    {
                        sessionOptions.AppendExecutionProvider_CUDA();
                        sp.GetRequiredService<ILogger<OnnxModelExecutorOptions>>().LogInformation("Using GPU accelerator");

                        sessionOptions.AppendExecutionProvider_CPU();
                    });
                    onnxOptions.ModelDir = options.ModelDir;
                })
            );

            localServices.AddSingleton(_ => TokenizationUtils.BERTTokenizerFromPretrained(options.ModelDir, options.TokenizerOptions));

            localServices.AddPipelineBuilder<TokenizedText, ClassificationResult<bool, float>>()
                .AddModelExecutor(sp =>
                {
                    var executorOptions = sp.GetRequiredService<IModelExecutorOptions>();
                    return ModelExecutorFactory.CreateModelExecutor(options.ModelExecutorType, executorOptions);
                })
                .AddInferenceSteps<TextClassification<bool>>()
                .AddBatchExecutor(s => new DecoratorChainBuilder(s)
                    .AddInitial<SerialPipelineBatchExecutor<TokenizedText, ClassificationResult<bool, float>>>()
                    .Decorate<MaxPaddedTokensBatchExecutor<TokenizedText, ClassificationResult<bool, float>>>()
                    .Decorate<TokenCountSortingBatchExecutor<TokenizedText, ClassificationResult<bool, float>>>()
                    .Build<IPipelineBatchExecutor<TokenizedText, ClassificationResult<bool, float>>>())
                .Build();

            localServices.AddSingleton<IInference<string, bool>, SentimentInference>();

            localServices.CopyToGlobal<IInference<string, bool>>();
        });
    }
}
