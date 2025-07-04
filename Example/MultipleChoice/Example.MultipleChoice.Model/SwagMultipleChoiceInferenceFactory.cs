using FAI.Core.Abstractions;
using FAI.Core.Configurations.PipelineBatchExecutors;
using FAI.Core.PipelineBatchExecutors;
using FAI.Core.ResultTypes;
using FAI.NLP.Configuration;
using FAI.NLP.Configuration.PipelineBatchExecutors;
using FAI.NLP.InferenceTasks.TextMultipleChoice;
using FAI.NLP.Tokenization;
using FAI.Onnx.Configuration;
using FAI.Onnx.Factories;
using FAI.Onnx.Utils;

namespace Example.MultipleChoice.Model;

using StreamedBatchExecutorBuilder =
    StreamedPipelineExecutorBuilder<TextMultipleChoiceInput, BatchTokenizedResult, ChoiceResult<TokenizedText>[], ChoiceResult<TokenizedText>>;

public static class SwagMultipleChoiceInferenceFactory
{
    public static async Task<IInference<SwagInput, ChoiceResult<TokenizedText>>> CreateMultipleChoiceInference(SwagMultipleChoiceInferenceOptions options)
    {
        Console.WriteLine($"Model: {options.ModelDir}");
        Func<Task<PretrainedTokenizer>> tokenizer = () => TokenizationUtils.BERTTokenizerFromPretrained(options.ModelDir, options.TokenizerOptions);
        Func<ValueTask<PretrainedTokenizer>> tokenizerFactory = tokenizer.GetTokenizerFactory();

        // MaxPaddedTokensBatchExecutorBuilder<TextMultipleChoiceInput, ChoiceResult<TokenizedText>> builder = CreateRoutedPipelineBatchExecutorBuilder(options, tokenizerFactory);
        var builder =
            new MaxPaddedTokensBatchExecutorBuilder<TextMultipleChoiceInput, ChoiceResult<TokenizedText>>
                {
                    MaxPaddedRatio = 0.1,
                    MaxTokensCount = 8192
                }
                .UseTokenizer(tokenizerFactory)
                .UseInnerPipelineExecutor<StreamedBatchExecutorBuilder>(builder =>
                    CreateGpuPipelineExecutionBuilder(builder, options, tokenizerFactory));


        var pipeline = new Pipeline<TextMultipleChoiceInput, ChoiceResult<TokenizedText>>(await builder.BuildAsync());

        return new SwagMultipleChoiceInference(pipeline);
    }

    private static MaxPaddedTokensBatchExecutorBuilder<TextMultipleChoiceInput, ChoiceResult<TokenizedText>> CreateRoutedPipelineBatchExecutorBuilder(
        SwagMultipleChoiceInferenceOptions options, Func<ValueTask<PretrainedTokenizer>> tokenizerFactory)
    {
        MaxPaddedTokensBatchExecutorBuilder<TextMultipleChoiceInput, ChoiceResult<TokenizedText>> builder =
            new MaxPaddedTokensBatchExecutorBuilder<TextMultipleChoiceInput, ChoiceResult<TokenizedText>>
                {
                    MaxPaddedRatio = 0.1,
                    MaxTokensCount = 8192
                }
                .UseTokenizer(tokenizerFactory)
                .UseInnerPipelineExecutor<RoutingPipelineExecutorBuilder<TextMultipleChoiceInput, ChoiceResult<TokenizedText>>>(routingBuilder =>
                {
                    routingBuilder.UseRoutingStrategy(new RoutingStrategy())
                        .UsePipelineExecutorBuilder<StreamedBatchExecutorBuilder>(builder =>
                            CreateGpuPipelineExecutionBuilder(builder, options, tokenizerFactory))
                        .UsePipelineExecutorBuilder<StreamedBatchExecutorBuilder>(builder =>
                            CreateCpuExecutorBuilder(builder, options, tokenizerFactory));
                });
        return builder;
    }

    private static void CreateGpuPipelineExecutionBuilder(
        StreamedBatchExecutorBuilder builder,
        SwagMultipleChoiceInferenceOptions options,
        Func<ValueTask<PretrainedTokenizer>> tokenizerFactory)
    {
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


        builder.MaxConcurrency = 4;
        builder.ParallelPreProcessing = false;
        builder.UseInferenceSteps<TextMultipleChoiceBuilder, TextMultipleChoiceTask>(classificationBuilder =>
        {
            classificationBuilder.MaxChoices = 4;
            classificationBuilder
                .UseTokenizer(tokenizerFactory)
                .UseModelExecutor(() => ModelExecutorFactory.CreateModelExecutor(options.ModelExecutorType, executorOptions));
        });
    }

    private static void CreateCpuExecutorBuilder(
        StreamedBatchExecutorBuilder builder,
        SwagMultipleChoiceInferenceOptions options,
        Func<ValueTask<PretrainedTokenizer>> tokenizerFactory)
    {
        var executorOptions = new OnnxModelExecutorOptions()
            .ConfigureOnnxOptions(onnxOptions =>
            {
                onnxOptions.ConfigureSessionOptions(sessionOptions =>
                {
                    sessionOptions.InterOpNumThreads = 2;
                    sessionOptions.IntraOpNumThreads = 2;
                    sessionOptions.AppendExecutionProvider_CPU();
                });
                onnxOptions.ModelDir = options.ModelDir;
            });

        builder.BatchSize = 5;
        builder.MaxConcurrency = 4;
        builder.UseInferenceSteps<TextMultipleChoiceBuilder, TextMultipleChoiceTask>(classificationBuilder =>
        {
            classificationBuilder.MaxChoices = 4;
            classificationBuilder
                .UseTokenizer(tokenizerFactory)
                .UseModelExecutor(() => ModelExecutorFactory.CreateModelExecutor(options.ModelExecutorType, executorOptions));
        });
    }
}

file class RoutingStrategy : IBatchExecutionRoutingStrategy<TextMultipleChoiceInput, ChoiceResult<TokenizedText>>
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