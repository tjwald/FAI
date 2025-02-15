using System.Numerics.Tensors;
using ML.Infra.Abstractions;
using ML.Infra.Configurations.PipelineBatchExecutors;
using ML.NLP.Configuration;
using ML.NLP.InferenceTasks;
using ML.NLP.PipelineBatchExecutors;
using ML.NLP.Tokenization;
using ML.Onnx.Factories;
using ML.Infra.PipelineBatchExecutors;
using ML.Infra.ResultTypes;

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

    private static IPipelineBatchExecutor<TInput, TOutput> CreatePipelineBatchExecutor<TInput, TPreprocess, TModelOutput, TOutput>(
        SentimentInferenceOptions options, IInferenceSteps<TInput, TOutput> inferenceSteps)
    {
        Console.WriteLine($"Using Model Pipeline Executor: {options.PipelineExecutorType}");
        IPipelineBatchExecutor<TInput, TOutput> executor = options.PipelineExecutorType switch
        {
            PipelineExecutorType.Serial => new SerialPipelineBatchExecutor<TInput, TOutput>(inferenceSteps, maxBatchSize: options.BatchSize),
            PipelineExecutorType.Parallel => new ParallelPipelineBatchExecutor<TInput, TOutput>(inferenceSteps, options.BatchSize, options.MaxConcurrency),
            PipelineExecutorType.Streamed => new StreamedBatchExecutor<TInput, TPreprocess, TModelOutput, TOutput>(
                inferenceSteps, options.BatchSize, options.MaxConcurrency, options.ParallelPreProcessing),
            _ => throw new ArgumentException("Unsupported pipeline executor type")
        };
        return executor;
    }

    private static SentimentInference CreateSentimentInference(SentimentInferenceOptions options, PretrainedTokenizer tokenizer,
        IModelExecutor<long, float> modelExecutor)
    {
        var textClassificationTask = new TextClassification<bool>(tokenizer, modelExecutor, new TextClassificationOptions<bool>([false, true]));

        IPipelineBatchExecutor<TokenizedText, ClassificationResult<bool>> executor =
            CreatePipelineBatchExecutor<TokenizedText, BatchTokenizedResult, Tensor<float>[], ClassificationResult<bool>>(options, textClassificationTask);

        if (options.UseTokenSortingExecution)
        {
            Console.WriteLine("Using out of order execution");
            executor = new TokenCountSortingBatchExecutor<ClassificationResult<bool>>(tokenizer, executor);
        }

        var pipeline = new Pipeline<TokenizedText, ClassificationResult<bool>>(executor);
        return new SentimentInference(pipeline);
    }
}