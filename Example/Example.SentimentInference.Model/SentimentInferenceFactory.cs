using System.Numerics.Tensors;
using ML.Infra.Abstractions;
using ML.Infra.Configurations.PipelineBatchExecutors;
using ML.Infra.Configurations.Pipelines;
using ML.Infra.Factories;
using ML.Infra.PipelineBatchExecutors;
using ML.Infra.Pipelines;
using ML.Infra.ResultTypes;
using ML.Infra.Tokenization;

namespace Example.SentimentInference.Model;

public static class SentimentInferenceFactory
{
    public static async Task<IInference<string, bool>> CreateSentimentInference(SentimentInferenceOptions options)
    {
        Console.WriteLine($"Model: {options.ModelDir}");
        var tokenizer = await TokenizationUtils.BERTTokenizerFromPretrained(options.ModelDir, options.TokenizerOptions);

        IModelExecutor<long, float> modelExecutor =
            await ModelExecutorFactory.CreateModelExecutor(options.ModelDir, options.ModelExecutorType, options.OnnxModelExecutorOptions);

        return CreateSentimentInference(options, tokenizer, modelExecutor);
    }

    private static IPipelineBatchExecutor<TInput, ClassificationResult<bool>> CreatePipelineBatchExecutor<TInput>(SentimentInferenceOptions options)
    {
        Console.WriteLine($"Using Model Pipeline Executor: {options.PipelineExecutorType}");
        IPipelineBatchExecutor<TInput, ClassificationResult<bool>> executor = options.PipelineExecutorType switch
        {
            PipelineExecutorType.Serial => new SerialPipelineBatchExecutor<TInput, ClassificationResult<bool>>(maxBatchSize: options.BatchSize),
            PipelineExecutorType.Parallel => new ParallelPipelineBatchExecutor<TInput, ClassificationResult<bool>>(options.BatchSize, options.MaxConcurrency),
            PipelineExecutorType.Streamed => new StreamedBatchExecutor<TInput, ClassificationResult<bool>, BatchTokenizedResult, Tensor<float>[]>(
                options.BatchSize, options.MaxConcurrency, options.ParallelPreProcessing),
            _ => throw new ArgumentException("Unsupported pipeline executor type")
        };
        return executor;
    }

    private static SentimentInferenceV2 CreateSentimentInference(SentimentInferenceOptions options, PretrainedTokenizer tokenizer,
        IModelExecutor<long, float> modelExecutor)
    {
        IPipelineBatchExecutor<TokenizedText, ClassificationResult<bool>> executor = CreatePipelineBatchExecutor<TokenizedText>(options);

        if (options.UseTokenSortingExecution)
        {
            Console.WriteLine("Using out of order execution");
            executor = new TokenCountSortingBatchExecutor<ClassificationResult<bool>>(tokenizer, executor);
        }

        var pipeline = new TextClassificationPipeline<bool>(tokenizer, modelExecutor, new TextClassificationOptions<bool>([false, true]), executor);
        return new SentimentInferenceV2(pipeline);
    }
}