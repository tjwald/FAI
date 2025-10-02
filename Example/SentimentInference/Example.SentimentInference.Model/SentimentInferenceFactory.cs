using FAI.Core.Abstractions;
using FAI.Core.Configurations.InferenceTasks;
using FAI.Core.Configurations.ModelExecutors;
using FAI.Core.PipelineBatchExecutors;
using FAI.Core.ResultTypes;
using FAI.Extensions.DependencyInjection.LocalServices;
using FAI.NLP.InferenceTasks.TextClassification;
using FAI.NLP.PipelineBatchExecutors;
using FAI.NLP.Tokenization;
using FAI.Onnx.Configuration;
using FAI.Onnx.Factories;
using Microsoft.Extensions.DependencyInjection;

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
                .AddConfigurationAndBind<MaxPaddedTokensBatchExecutorOptions>("SentimentInference:BatchExecutors:MaxPaddedTokens");

            localServices.AddSingleton<IModelExecutorConfig, OnnxModelExecutorOptions>(_ => new OnnxModelExecutorOptions()
                .ConfigureOnnxOptions(onnxOptions =>
                {
                    onnxOptions.ConfigureSessionOptions(sessionOptions =>
                    {
                        sessionOptions.AppendExecutionProvider_CUDA();
                        Console.WriteLine("Using GPU accelerator");

                        sessionOptions.AppendExecutionProvider_CPU();
                    });
                    onnxOptions.ModelDir = options.ModelDir;
                })
            );

            localServices.AddSingleton(_ => TokenizationUtils.BERTTokenizerFromPretrained(options.ModelDir, options.TokenizerOptions));

            localServices.AddSentimentInference()
                .AddModelExecutor(sp =>
                {
                    var executorOptions = sp.GetRequiredService<IModelExecutorConfig>();
                    return ModelExecutorFactory.CreateModelExecutor(options.ModelExecutorType, executorOptions);
                })
                .AddInferenceSteps<TextClassification<bool>>()
                .AddBatchExecutor(s =>
                {
                    new DecoratorChainBuilder(s)
                        .AddInitial<SerialPipelineBatchExecutor<TokenizedText, ClassificationResult<bool, float>>>()
                        .Decorate<MaxPaddedTokensBatchExecutor<TokenizedText, ClassificationResult<bool, float>>>()
                        .Decorate<TokenCountSortingBatchExecutor<TokenizedText, ClassificationResult<bool, float>>>()
                        .Build<IPipelineBatchExecutor<TokenizedText, ClassificationResult<bool, float>>>();
                })
                .Build();

            localServices.CopyToGlobal<IInference<string, bool>>();
        });
    }
}
