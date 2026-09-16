using System.Numerics.Tensors;
using FAI.Core.Abstractions;
using FAI.Core.Configurations;
using FAI.Core.Configurations.ModelExecutors;
using FAI.Core.Extensions.DI;
using FAI.Core.Pipelines;
using FAI.NLP.Extensions.DI;
using FAI.NLP.InferenceTasks.TextClassification;
using FAI.NLP.InferenceTasks.TextEmbedding;
using FAI.NLP.Pipelines;
using FAI.NLP.Tokenization;
using FAI.Onnx;
using FAI.Onnx.Configuration;
using Microsoft.Extensions.DependencyInjection;

namespace Example.TextEmbedding.Model;

public static class TextEmbeddingFactory
{
    public static IServiceCollection AddTextEmbeddingInference(
        this IServiceCollection services,
        TextEmbeddingModelOptions options)
    {
        return services.AddLocalServices(localServices =>
        {
            localServices.AddSingleton<IModelExecutorOptions>(new OnnxModelExecutorOptions()
                .ConfigureOnnxOptions(onnxOptions =>
                {
                    onnxOptions.ConfigureSessionOptions(sessionOptions =>
                    {
                        if (options.UseGpu)
                        {
                            sessionOptions.AppendExecutionProvider_CUDA();
                        }

                        sessionOptions.AppendExecutionProvider_CPU();
                    });
                    onnxOptions.ModelDir = options.ModelDirectory;
                    onnxOptions.ModelFileName = "model.onnx";
                }));
            localServices.AddSingleton(options.DecodingOptions);
            localServices.AddSingleton(_ => TokenizationUtils.BERTTokenizerFromPretrained(options.ModelDirectory, options.TokenizerOptions));
            localServices.AddSingleton(options.TokenCountOrdering);
            localServices.AddSingleton(options.MaxPaddedTokens);
            localServices.AddSingleton(options.ParallelScheduler);
            localServices.AddSingleton<IPartitionScheduler>(serviceProvider =>
                new ParallelPartitionScheduler(serviceProvider.GetRequiredService<ParallelPartitionSchedulerOptions>()));
            localServices.AddSingleton<TextTensorization>();
            localServices.AddSingleton<TextEmbeddingDecoding>();
            localServices.AddReadOnlyMemoryBatch<string>();
            localServices.AddTensorBatch<float>();

            localServices
                .AddPipeline<ReadOnlyMemory<string>>()
                .UseTensorDestinationAllocation<ReadOnlyMemory<string>, float>(
                    options.DecodingOptions.EmbeddingDimensions ?? TextEmbeddingDecoding.DefaultEmbeddingDimensions)
                .Then<ReadOnlyMemory<TokenizedText>, TextTokenization>()
                .UseTokenCountOrdering()
                .UseMaxPaddedTokensPartitioning()
                .Fork(inner => inner
                    .Then<NamedTensorCollection, TextTensorization>()
                    .ThenOnnxModel())
                .Then<Tensor<float>, TextEmbeddingDecoding>()
                .Build();

            localServices.AddPipelineInference<ReadOnlyMemory<string>, Tensor<float>>();
            localServices.CopyToGlobal<IInference<ReadOnlyMemory<string>, Tensor<float>>>();
        });
    }
}
