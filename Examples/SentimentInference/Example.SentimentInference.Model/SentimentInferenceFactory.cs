using System.Numerics.Tensors;
using FAI.Core.Abstractions;
using FAI.Core.Configurations;
using FAI.Core.Configurations.InferenceTasks;
using FAI.Core.Configurations.ModelExecutors;
using FAI.Core.Extensions.DI;
using FAI.Core.ResultTypes;
using FAI.Core.Steps;
using FAI.NLP.Configuration;
using FAI.NLP.Extensions.DI;
using FAI.NLP.InferenceTasks.TextClassification;
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

            localServices.AddSingleton(_ => TokenizationUtils.BERTTokenizerFromPretrained(options.ModelDir, options.TokenizerOptions));
            localServices.AddConfigurationAndBind<ClassificationOptions<bool>>("SentimentInference:Classification");
            localServices.AddConfigurationAndBind<TokenCountOrderingOptions>("SentimentInference:BatchExecutors:TokenCountSorting");
            localServices.AddConfigurationAndBind<MaxPaddedTokensPartitionerOptions>("SentimentInference:BatchExecutors:MaxPaddedTokens");
            localServices.AddConfigurationAndBind<ParallelPartitionSchedulerOptions>("SentimentInference:BatchExecutors:ParallelSchedular");
            localServices.AddSingleton<IPartitionScheduler>(sp =>
                new ParallelPartitionScheduler(sp.GetRequiredService<ParallelPartitionSchedulerOptions>()));
            localServices.AddSingleton(sp =>
                ModelExecutorFactory.CreateBorrowedModelStep(
                    options.ModelExecutorType,
                    sp.GetRequiredService<IModelExecutorOptions>()));
            localServices.AddSingleton<ClassificationDecodingStep<bool>>();

            localServices
                .AddPipeline<ReadOnlyMemory<TokenizedText>>()
                .Then(
                    pipeline => pipeline
                        .Then<Tensor<long>[], TextBatchEncodingStep>()
                        .ThenBorrowed(
                            sp => sp.GetRequiredService<IBorrowedTensorProducer<Tensor<long>[], float>>(),
                            sp => sp.GetRequiredService<ClassificationDecodingStep<bool>>(),
                            (_, input, cancellationToken) =>
                            {
                                cancellationToken.ThrowIfCancellationRequested();
                                int batchSize = checked((int)input[0].Lengths[0]);
                                var output = new ClassificationResult<bool, float>[batchSize];
                                return ValueTask.FromResult(
                                    new BatchLease<Memory<ClassificationResult<bool, float>>>(output));
                            }),
                    (_, input, cancellationToken) =>
                    {
                        cancellationToken.ThrowIfCancellationRequested();
                        var output = new ClassificationResult<bool, float>[input.Length];
                        return ValueTask.FromResult(new BatchLease<Memory<ClassificationResult<bool, float>>>(output));
                    },
                    stage => stage
                        .UseTokenizingStep()
                        .UseTokenCountOrderingStep()
                        .UseMaxPaddedTokensPartitioningStep())
                .Build();

            localServices.AddSingleton<IInference<string, bool>, SentimentInference>();

            localServices.CopyToGlobal<IInference<string, bool>>();
        });
    }
}
