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
    public static async Task<SentimentInference> CreateSentimentInference(SentimentInferenceOptions options)
    {
        var tokenizer = await TokenizationUtils.BERTTokenizerFromPretrained(options.ModelDir, options.TokenizerOptions);

        IModelExecutor<long, float> modelExecutor =
            await ModelExecutorFactory.CreateModelExecutor(options.ModelDir, options.ModelExecutorType, options.OnnxModelExecutorOptions);

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
            executor = new OutOfOrderBatchExecutor<ClassificationResult<bool>>(tokenizer.Tokenizer, executor);
        }

        var pipeline = new TextClassificationPipeline<bool>(tokenizer, modelExecutor, new TextClassificationOptions<bool>([false, true]), executor);
        return new SentimentInference(pipeline);
    }
}