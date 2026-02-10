using FAI.Core.Abstractions;
using FAI.Core.Configurations.ModelExecutors;
using FAI.Core.Extensions.DI;
using FAI.Core.ResultTypes;
using FAI.NLP.Extensions.DI;
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

            localServices.AddPipeline<TokenizedText, ClassificationResult<bool, float>>()
                .AddModelExecutor(sp =>
                {
                    var executorOptions = sp.GetRequiredService<IModelExecutorOptions>();
                    return ModelExecutorFactory.CreateModelExecutor(options.ModelExecutorType, executorOptions);
                })
                .WithTextClassification(section: "SentimentInference:Classification")
                .UseTokenSorting(section: "SentimentInference:BatchExecutors:TokenCountSorting")
                .UsePartitioning(partitionBuilder =>
                    partitionBuilder
                        .WithMaxPaddedTokens(section: "SentimentInference:BatchExecutors:MaxPaddedTokens")
                        .WithParallelSchedular(section: "SentimentInference:BatchExecutors:ParallelSchedular")
                );

            localServices.AddSingleton<IInference<string, bool>, SentimentInference>();

            localServices.CopyToGlobal<IInference<string, bool>>();
        });
    }
}
