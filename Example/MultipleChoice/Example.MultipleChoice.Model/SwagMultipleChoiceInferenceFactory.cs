using FAI.Core.Abstractions;
using FAI.Core.Configurations.ModelExecutors;
using FAI.Core.Configurations.PipelineBatchExecutors;
using FAI.Core.Extensions.DI;
using FAI.Core.PipelineBatchExecutors;
using FAI.Core.ResultTypes;
using FAI.Extensions.DependencyInjection.LocalServices;
using FAI.NLP.Configuration;
using FAI.NLP.Configuration.PipelineBatchExecutors;
using FAI.NLP.InferenceTasks.TextMultipleChoice;
using FAI.NLP.PipelineBatchExecutors;
using FAI.NLP.Tokenization;
using FAI.Onnx.Configuration;
using FAI.Onnx.Factories;
using FAI.Onnx.Utils;
using Microsoft.Extensions.DependencyInjection;

namespace Example.MultipleChoice.Model;

public static class SwagMultipleChoiceInferenceFactory
{
    public static IServiceCollection AddDefaultSwagInference(this IServiceCollection services, SwagMultipleChoiceInferenceOptions options)
    {
        services
            .AddConfigurationAndBind<TextMultipleChoiceOptions>("SwagInference:MultipleChoice")
            .AddConfigurationAndBind<StreamedPipelineExecutorOptions>("SwagInference:BatchExecutors:Streamed")
            .AddConfigurationAndBind<MaxPaddedTokensBatchExecutorOptions>("SwagInference:BatchExecutors:MaxPaddedTokens");

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
        services.AddSingleton<InferenceSteps<TextMultipleChoiceInput, BatchTokenizedResult, ChoiceResult<TokenizedText>[], ChoiceResult<TokenizedText>>, TextMultipleChoiceTask>();

        services.AddPipelineBuilder<TextMultipleChoiceInput, ChoiceResult<TokenizedText>>()
            .AddModelExecutor(sp =>
            {
                var executorOptions = sp.GetRequiredService<IModelExecutorOptions>();
                return ModelExecutorFactory.CreateModelExecutor(options.ModelExecutorType, executorOptions);
            })
            .AddBatchExecutor(s => new DecoratorChainBuilder(s)
                .AddInitial<StreamedBatchExecutor<TextMultipleChoiceInput, BatchTokenizedResult, ChoiceResult<TokenizedText>[], ChoiceResult<TokenizedText>>>()
                .Decorate<MaxPaddedTokensBatchExecutor<TextMultipleChoiceInput, ChoiceResult<TokenizedText>>>()
                .Decorate<TokenCountSortingBatchExecutor<TextMultipleChoiceInput, ChoiceResult<TokenizedText>>>()
                .Build<IPipelineBatchExecutor<TextMultipleChoiceInput, ChoiceResult<TokenizedText>>>())
            .Build();

        services.AddSingleton<IInference<SwagInput, ChoiceResult<TokenizedText>>, SwagMultipleChoiceInference>();

        return services;
    }

    public static IServiceCollection AddRoutedSwagInference(this IServiceCollection services, SwagMultipleChoiceInferenceOptions options)
    {
        services
            .AddConfigurationAndBind<TextMultipleChoiceOptions>("SwagInference:MultipleChoice")
            .AddConfigurationAndBind<StreamedPipelineExecutorOptions>("SwagInference:BatchExecutors:Streamed")
            .AddConfigurationAndBind<MaxPaddedTokensBatchExecutorOptions>("SwagInference:BatchExecutors:MaxPaddedTokens");

        services.AddSingleton(_ => TokenizationUtils.BERTTokenizerFromPretrained(options.ModelDir, options.TokenizerOptions));

        services.AddLocalServices(gpuServices =>
        {
            gpuServices.AddSingleton<IModelExecutorOptions, OnnxModelExecutorOptions>(_ => new OnnxModelExecutorOptions()
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

            gpuServices.AddSingleton(sp =>
            {
                var executorOptions = sp.GetRequiredService<IModelExecutorOptions>();
                return ModelExecutorFactory.CreateModelExecutor(options.ModelExecutorType, executorOptions);
            });

            gpuServices.AddDecoratedChain()
                .AddInitial<ParallelPipelineBatchExecutor<TextMultipleChoiceInput, ChoiceResult<TokenizedText>>>()
                .Decorate<MaxPaddedTokensBatchExecutor<TextMultipleChoiceInput, ChoiceResult<TokenizedText>>>()
                .RegisterAs<IPipelineBatchExecutor<TextMultipleChoiceInput, ChoiceResult<TokenizedText>>>();

            gpuServices.CopyToGlobal<IPipelineBatchExecutor<TextMultipleChoiceInput, ChoiceResult<TokenizedText>>>();
        });

        services.AddLocalServices(cpuServices =>
        {
            cpuServices.AddSingleton(new StreamedPipelineExecutorOptions(5, 4));
            cpuServices.AddSingleton<IModelExecutorOptions, OnnxModelExecutorOptions>(_ => new OnnxModelExecutorOptions()
                .ConfigureOnnxOptions(onnxOptions =>
                {
                    onnxOptions.ConfigureSessionOptions(sessionOptions =>
                    {
                        sessionOptions.InterOpNumThreads = 2;
                        sessionOptions.IntraOpNumThreads = 2;
                        sessionOptions.AppendExecutionProvider_CPU();
                    });
                    onnxOptions.ModelDir = options.ModelDir;
                }));

            cpuServices.AddSingleton(sp =>
            {
                var executorOptions = sp.GetRequiredService<IModelExecutorOptions>();
                return ModelExecutorFactory.CreateModelExecutor(options.ModelExecutorType, executorOptions);
            });

            cpuServices
                .AddSingleton<IPipelineBatchExecutor<TextMultipleChoiceInput, ChoiceResult<TokenizedText>>,
                    StreamedBatchExecutor<TextMultipleChoiceInput, BatchTokenizedResult, ChoiceResult<TokenizedText>[], ChoiceResult<TokenizedText>>>();

            cpuServices.CopyToGlobal<IPipelineBatchExecutor<TextMultipleChoiceInput, ChoiceResult<TokenizedText>>>();
        });


        services.AddPipelineBuilder<TextMultipleChoiceInput, ChoiceResult<TokenizedText>>()
            .AddInferenceSteps<TextMultipleChoiceTask>()
            .AddBatchExecutor((IServiceProvider sp) =>
            {
                var executors = sp.GetServices<IPipelineBatchExecutor<TextMultipleChoiceInput, ChoiceResult<TokenizedText>>>().ToArray();

                return new RoutingPipelineBatchExecutor<TextMultipleChoiceInput, ChoiceResult<TokenizedText>>(executors, new RoutingStrategy());
            })
            .Build();

        services.AddSingleton<IInference<SwagInput, ChoiceResult<TokenizedText>>, SwagMultipleChoiceInference>();

        return services;
    }
}

file sealed class RoutingStrategy : IBatchExecutionRoutingStrategy<TextMultipleChoiceInput, ChoiceResult<TokenizedText>>
{
    private readonly CircularAtomicCounter _clock = new(50);

    public List<BatchExecutionRoutingResult<TextMultipleChoiceInput, ChoiceResult<TokenizedText>>> Route(
        IPipelineBatchExecutor<TextMultipleChoiceInput, ChoiceResult<TokenizedText>>[] executors, ReadOnlyMemory<TextMultipleChoiceInput> inputs)
    {
        var next = _clock.Next();
        int executorIndex = next > 0 ? 0 : 1;
        return
        [
            new(executors[executorIndex], [new(0, inputs.Length)]),
        ];
    }
}
