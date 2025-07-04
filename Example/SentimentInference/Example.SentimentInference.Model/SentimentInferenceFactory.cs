using FAI.Core.Abstractions;
using FAI.Core.Configurations.PipelineBatchExecutors;
using FAI.Core.ResultTypes;
using FAI.NLP.Configuration;
using FAI.NLP.Configuration.PipelineBatchExecutors;
using FAI.NLP.InferenceTasks.TextClassification;
using FAI.NLP.Tokenization;
using FAI.Onnx.Configuration;
using FAI.Onnx.Factories;

namespace Example.SentimentInference.Model;

using StreamedBatchExecutor = StreamedPipelineExecutorBuilder<TokenizedText, BatchTokenizedResult, ClassificationResult<bool>[], ClassificationResult<bool>>;

public static class SentimentInferenceFactory
{
    public static async Task<IInference<string, bool>> CreateSentimentInference(SentimentInferenceOptions options)
    {
        Console.WriteLine($"Model: {options.ModelDir}");
        Func<Task<PretrainedTokenizer>> tokenizer = () => TokenizationUtils.BERTTokenizerFromPretrained(options.ModelDir, options.TokenizerOptions);
        Func<ValueTask<PretrainedTokenizer>> tokenizerFactory = tokenizer.GetTokenizerFactory();

        var executorBuilder = CreateBatchGpuExecutorBuilder(options, tokenizerFactory);

        var pipeline = new Pipeline<TokenizedText, ClassificationResult<bool>>(await executorBuilder.BuildAsync());
        return new SentimentInference(pipeline);
    }

    private static MaxPaddedTokensBatchExecutorBuilder<TokenizedText, ClassificationResult<bool>> CreateBatchGpuExecutorBuilder(
        SentimentInferenceOptions options,
        Func<ValueTask<PretrainedTokenizer>> tokenizerFactory)
    {
        MaxPaddedTokensBatchExecutorBuilder<TokenizedText, ClassificationResult<bool>> builder = new();
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
            .UseInnerPipelineExecutor<StreamedBatchExecutor>(builder =>
            {
                builder.MaxConcurrency = 4;
                builder.ParallelPreProcessing = false;
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