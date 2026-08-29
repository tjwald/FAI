using FAI.Core.Abstractions;
using FAI.Core.Configurations;
using FAI.Core.Configurations.ModelExecutors;
using FAI.Core.Extensions.DI;
using FAI.Core.Pipelines;
using FAI.Core.ResultTypes;
using FAI.NLP.Configuration;
using FAI.NLP.Extensions.DI;
using FAI.NLP.InferenceTasks.TextMultipleChoice;
using FAI.NLP.Pipelines;
using FAI.NLP.Tokenization;
using FAI.Onnx.Configuration;
using FAI.Onnx.Factories;
using Microsoft.Extensions.DependencyInjection;

namespace Example.MultipleChoice.Model;

public static class SwagMultipleChoiceInferenceFactory
{
    public static IServiceCollection AddDefaultSwagInference(this IServiceCollection services, SwagMultipleChoiceInferenceOptions options)
    {
        return services.AddLocalServices(localServices =>
        {
            localServices.AddConfigurationAndBind<TextMultipleChoiceOptions>("SwagInference:MultipleChoice");
            localServices.AddConfigurationAndBind<TokenCountOrderingOptions>("SwagInference:BatchExecutors:TokenCountSorting");
            localServices.AddConfigurationAndBind<MaxPaddedTokensPartitionerOptions>("SwagInference:BatchExecutors:MaxPaddedTokens");
            localServices.AddConfigurationAndBind<ParallelPartitionSchedulerOptions>("SwagInference:BatchExecutors:Parallel");
            localServices.AddSingleton<IPartitionScheduler>(sp =>
                new ParallelPartitionScheduler(sp.GetRequiredService<ParallelPartitionSchedulerOptions>()));

            localServices.AddSingleton<IModelExecutorOptions, OnnxModelExecutorOptions>(_ => new OnnxModelExecutorOptions()
                .ConfigureOnnxOptions(onnxOptions =>
                {
                    onnxOptions.ConfigureSessionOptions(sessionOptions =>
                    {
                        if (options.UseGpu)
                        {
                            sessionOptions.AppendExecutionProvider_CUDA();
                            Console.WriteLine("Using GPU accelerator");
                        }

                        sessionOptions.AppendExecutionProvider_CPU();
                    });
                    onnxOptions.ModelDir = options.ModelDir;
                })
            );
            localServices.AddSingleton(_ => TokenizationUtils.BERTTokenizerFromPretrained(options.ModelDir, options.TokenizerOptions));
            localServices.AddSingleton(sp =>
                ModelExecutorFactory.CreateModelPipeline(
                    options.ModelExecutorType,
                    sp.GetRequiredService<IModelExecutorOptions>()));

            localServices
                .AddPipeline<ReadOnlyMemory<TextMultipleChoiceInput>>()
                .Then<ReadOnlyMemory<TokenizedTextMultipleChoiceInput>, TextMultipleChoiceTokenization>()
                .UseTokenCountOrdering()
                .UseMaxPaddedTokensPartitioning()
                .Then<Memory<ChoiceResult<TokenizedText>>, TextMultipleChoicePipeline>()
                .Build();

            localServices.AddSingleton<IInference<SwagInput, ChoiceResult<TokenizedText>>, SwagMultipleChoiceInference>();
            localServices.CopyToGlobal<IInference<SwagInput, ChoiceResult<TokenizedText>>>();
        });
    }
}
