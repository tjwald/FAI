using System.Numerics.Tensors;
using FAI.Core.Abstractions;
using FAI.Core.Configurations;
using FAI.Core.Configurations.InferenceTasks;
using FAI.Core.Configurations.ModelExecutors;
using FAI.Core.Extensions.DI;
using FAI.Core.Pipelines;
using FAI.Core.ResultTypes;
using FAI.NLP.Configuration;
using FAI.NLP.Extensions.DI;
using FAI.NLP.InferenceTasks.TextClassification;
using FAI.NLP.Pipelines;
using FAI.NLP.Tokenization;
using FAI.Onnx;
using FAI.Onnx.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;

namespace Example.SentimentInference.Model;

public static class SentimentInferenceFactory
{
    public static IServiceCollection AddDefaultSentimentInference(this IServiceCollection services, SentimentInferenceOptions options)
    {
        return services.AddLocalServices(localServices =>
        {
            localServices.AddSingleton(sp => new OnnxModelExecutorOptions()
                .ConfigureOnnxOptions(onnxOptions =>
                {
                    onnxOptions.ConfigureSessionOptions(sessionOptions =>
                    {
                        if (options.UseGpu)
                        {
                            sessionOptions.AppendExecutionProvider_CUDA();
                            sp.GetRequiredService<ILogger<OnnxModelExecutorOptions>>().LogInformation("Using GPU accelerator");
                        }

                        sessionOptions.AppendExecutionProvider_CPU();
                    });
                    onnxOptions.ModelDir = options.ModelDir;
                })
            );
            localServices.AddSingleton<IModelExecutorOptions>(sp => sp.GetRequiredService<OnnxModelExecutorOptions>());

            localServices.AddSingleton(_ => TokenizationUtils.BERTTokenizerFromPretrained(options.ModelDir, options.TokenizerOptions));
            localServices.AddConfigurationAndBind<ClassificationOptions<bool>>("SentimentInference:Classification");
            localServices.AddConfigurationAndBind<TokenCountOrderingOptions>("SentimentInference:BatchExecutors:TokenCountSorting");
            localServices.AddConfigurationAndBind<MaxPaddedTokensPartitionerOptions>("SentimentInference:BatchExecutors:MaxPaddedTokens");
            localServices.AddConfigurationAndBind<ParallelPartitionSchedulerOptions>("SentimentInference:BatchExecutors:ParallelSchedular");
            localServices.AddSingleton<IPartitionScheduler>(sp =>
                new ParallelPartitionScheduler(sp.GetRequiredService<ParallelPartitionSchedulerOptions>()));
            localServices.AddSingleton<ClassificationDecoding<bool>>();

            localServices
                .AddPipeline<ReadOnlyMemory<string>>()
                .Then<ReadOnlyMemory<TokenizedText>, TextTokenization>()
                .UseTokenCountOrdering()
                .UseMaxPaddedTokensPartitioning()
                .Then<Tensor<long>[], TextTensorization>()
                .ThenOnnxModel()
                .Then<Memory<ClassificationResult<bool, float>>, ClassificationDecoding<bool>>()
                .Build();

            localServices.AddSingleton<IInference<string, bool>, SentimentInference>();

            localServices.CopyToGlobal<IInference<string, bool>>();
        });
    }
}
