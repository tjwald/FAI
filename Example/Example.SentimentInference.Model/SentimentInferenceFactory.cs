using System.Numerics.Tensors;
using ML.Infra.Abstractions;
using ML.Infra.Configurations.PipelineBatchExecutors;
using ML.Infra.Factories;
using ML.Infra.PipelineBatchExecutors;
using ML.Infra.Pipelines;
using ML.Infra.Tokenization;

namespace Example.SentimentInference.Model;

public static class SentimentInferenceFactory
{
    public static async Task<SentimentInference> CreateSentimentInference(SentimentInferenceOptions options, bool useRaw)
    {
        if (useRaw)
        {
            return await CreateRawPipeline(options);
        }
        
        Console.WriteLine($"Model: {options.ModelDir}");
        var tokenizer = await TokenizationUtils.BERTTokenizerFromPretrained(options.ModelDir, options.TokenizerOptions);

        IModelExecutor<long, float> modelExecutor =
            await ModelExecutorFactory.CreateModelExecutor(options.ModelDir, options.ModelExecutorType, options.OnnxModelExecutorOptions);

        Console.WriteLine($"Using Model Pipeline Executor: {options.PipelineExecutorType}");
        IPipelineBatchExecutor<string, ClassificationResult<bool>> executor = options.PipelineExecutorType switch
        {
            PipelineExecutorType.Serial => new SerialPipelineBatchExecutor<string, ClassificationResult<bool>>(maxBatchSize: options.BatchSize),
            PipelineExecutorType.Parallel => new ParallelPipelineBatchExecutor<string, ClassificationResult<bool>>(options.BatchSize, options.MaxConcurrency),
            PipelineExecutorType.Streamed => new StreamedBatchExecutor<string, ClassificationResult<bool>, BatchTokenizedResult, Tensor<float>[]>(
                options.BatchSize, options.MaxConcurrency, options.ParallelPreProcessing),
            _ => throw new ArgumentException("Unsupported pipeline executor type")
        };

        if (options.UseOutOfOrderExecution)
        {
            Console.WriteLine("Using out of order execution");
            executor = new OutOfOrderBatchExecutor<ClassificationResult<bool>>(tokenizer.Tokenizer, executor);
        }

        var pipeline = new TextClassificationPipeline<bool>(tokenizer, modelExecutor, new TextClassificationOptions<bool>([false, true]), executor);
        return new SentimentInference(pipeline);
    }

    private static async Task<SentimentInference> CreateRawPipeline(SentimentInferenceOptions options)
    {
        var tokenizer = await TokenizationUtils.BERTTokenizerFromPretrained(options.ModelDir, options.TokenizerOptions);
        IModelExecutor<long, float> modelExecutor =
            await ModelExecutorFactory.CreateModelExecutor(options.ModelDir, options.ModelExecutorType, options.OnnxModelExecutorOptions);

        RawTextClassificationPipeline<bool> pipeline =
            new RawTextClassificationPipeline<bool>(tokenizer.Tokenizer, options.TokenizerOptions, modelExecutor, [false, true], options.BatchSize, options.MaxConcurrency);

        return new SentimentInference(pipeline);
    }
}