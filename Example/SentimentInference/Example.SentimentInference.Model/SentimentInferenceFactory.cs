using FAI.Core.Abstractions;
using FAI.Core.Configurations.InferenceTasks;
using FAI.Core.Configurations.ModelExecutors;
using FAI.Core.Configurations.PipelineBatchExecutors;
using FAI.Core.PipelineBatchExecutors;
using FAI.Core.ResultTypes;
using FAI.NLP.Configuration;
using FAI.NLP.Configuration.PipelineBatchExecutors;
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
        services
           .AddConfigurationAndBind<ClassificationOptions<bool>>("SentimentInference:Classification")
           .AddConfigurationAndBind<SerialPipelineBatchExecutorOptions>("SentimentInference:BatchExecutors:SerialPipeline")
           .AddConfigurationAndBind<MaxPaddedTokensBatchExecutorOptions>("SentimentInference:BatchExecutors:MaxPaddedTokens");

        services.AddSentimentInference()
            .AddTokenizer(_ => TokenizationUtils.BERTTokenizerFromPretrained(options.ModelDir, options.TokenizerOptions))
            .AddLocal<IModelExecutorConfig, OnnxModelExecutorOptions>(new OnnxModelExecutorOptions()
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
            ).AddModelExecutor(sp =>
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

        return services;
    }

    private static IPipelineBatchExecutorBuilder<TokenizedText, ClassificationResult<bool, float>> CreateBatchGpuExecutorBuilder(
        SentimentInferenceOptions options,
        PretrainedTokenizer tokenizerFactory)
    {
        MaxPaddedTokensBatchExecutorBuilder<TokenizedText, ClassificationResult<bool, float>> builder = new();
        var executorOptions = new OnnxModelExecutorOptions()
            .ConfigureOnnxOptions(onnxOptions =>
            {
                onnxOptions.ConfigureSessionOptions(sessionOptions =>
                {
                    sessionOptions.AppendExecutionProvider_CUDA();
                    Console.WriteLine("Using GPU accelerator");

                    sessionOptions.AppendExecutionProvider_CPU();
                });
                onnxOptions.ModelDir = options.ModelDir;
            });

        builder.MaxPaddedRatio = 0.1;
        builder.MaxTokensCount = 2048;

        builder
            .UseTokenizer(tokenizerFactory)
            .UseInnerPipelineExecutor<SerialPipelineExecutorBuilder<TokenizedText, ClassificationResult<bool, float>>>(builder =>
            {
                builder.UseInferenceSteps<TextClassificationBuilder<bool>, TextClassification<bool>>(classificationBuilder =>
                {
                    classificationBuilder
                        .UseChoices(false, true)
                        .UseTokenizer(tokenizerFactory)
                        .UseModelExecutor(() => ModelExecutorFactory.CreateModelExecutor(options.ModelExecutorType, executorOptions));
                });
            });

        return builder;
    }
}
