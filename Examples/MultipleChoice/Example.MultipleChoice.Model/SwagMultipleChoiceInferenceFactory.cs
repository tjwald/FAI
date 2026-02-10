using FAI.Core.Abstractions;
using FAI.Core.Configurations.ModelExecutors;
using FAI.Core.Configurations.PipelineBatchExecutors;
using FAI.Core.Extensions.DI;
using FAI.Core.ResultTypes;
using FAI.NLP.Configuration;
using FAI.NLP.Extensions.DI;
using FAI.NLP.InferenceTasks.TextMultipleChoice;
using FAI.NLP.Tokenization;
using FAI.Onnx.Configuration;
using FAI.Onnx.Factories;
using Microsoft.Extensions.DependencyInjection;

namespace Example.MultipleChoice.Model;

public static class SwagMultipleChoiceInferenceFactory
{
    public static IServiceCollection AddDefaultSwagInference(this IServiceCollection services, SwagMultipleChoiceInferenceOptions options)
    {
        services
            .AddConfigurationAndBind<TextMultipleChoiceOptions>("SwagInference:MultipleChoice")
            .AddConfigurationAndBind<StreamedPipelineExecutorOptions>("SwagInference:BatchExecutors:Streamed");

        services.AddSingleton<IModelExecutorOptions, OnnxModelExecutorOptions>(_ => new OnnxModelExecutorOptions()
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

        services.AddSingleton(_ => TokenizationUtils.BERTTokenizerFromPretrained(options.ModelDir, options.TokenizerOptions));
        services
            .AddSingleton<IInferenceSteps<TextMultipleChoiceInput, ChoiceResult<TokenizedText>>, TextMultipleChoiceTask>();

        services.AddPipeline<TextMultipleChoiceInput, ChoiceResult<TokenizedText>>()
            .AddModelExecutor(sp =>
            {
                var executorOptions = sp.GetRequiredService<IModelExecutorOptions>();
                return ModelExecutorFactory.CreateModelExecutor(options.ModelExecutorType, executorOptions);
            })
            .UseTokenSorting()
            .UsePartitioning(partitionBuilder =>
            {
                partitionBuilder.WithMaxPaddedTokens(section: "SwagInference:BatchExecutors:MaxPaddedTokens")
                    .WithParallelSchedular("SwagInference:BatchExecutors:Parallel"); // TODO
            });

        services.AddSingleton<IInference<SwagInput, ChoiceResult<TokenizedText>>, SwagMultipleChoiceInference>();

        return services;
    }
}
