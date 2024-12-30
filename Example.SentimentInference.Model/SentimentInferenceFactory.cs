using ML.Infra.Abstractions;
using ML.Infra.ModelExecutors;
using ML.Infra.ModelExecutors.Onnx;
using ML.Infra.PipelineBatchExecutors;
using ML.Infra.Pipelines;
using ML.Infra.Tokenization;

namespace Example.SentimentInference.Model;

public enum ModelExecutorType
{
    Simple,
    Pooled,
    Async,
    AsyncPooled,
}

public record SentimentInferenceOptions(
    string ModelDir,
    PretrainedTokenizerOptions TokenizerOptions,
    OnnxModelExecutorOptions OnnxModelExecutorOptions,
    int? MaxConcurrency,
    int BatchSize,
    bool UseOutOfOrderExecution,
    ModelExecutorType ModelExecutorType);

public static class SentimentInferenceFactory
{
    public static async Task<SentimentInference> CreateSentimentInference(SentimentInferenceOptions options)
    {
        var tokenizer = await TokenizationUtils.BERTTokenizerFromPretrained(options.ModelDir, options.TokenizerOptions);

        IModelExecutor<long, float> modelExecutor = options.ModelExecutorType switch
        {
            ModelExecutorType.Simple => await OnnxModelExecutor.FromPretrained(options.ModelDir, options.OnnxModelExecutorOptions),
            ModelExecutorType.Pooled => new PooledModelExecutor<long, float>(new OnnxModelExecutorObjectPool<OnnxModelExecutor>(options.ModelDir,
                options.OnnxModelExecutorOptions)),
            ModelExecutorType.Async => await AsyncOnnxModelExecutor.FromPretrained(options.ModelDir, options.OnnxModelExecutorOptions),
            ModelExecutorType.AsyncPooled => new PooledModelExecutor<long, float>(new OnnxModelExecutorObjectPool<AsyncOnnxModelExecutor>(options.ModelDir,
                options.OnnxModelExecutorOptions)),
            _ => throw new ArgumentOutOfRangeException(),
        };

        IPipelineBatchExecutor<string, ClassificationResult<bool>> executor =
            new ParallelPipelineBatchExecutor<string, ClassificationResult<bool>>(options.BatchSize, options.MaxConcurrency);
        if (options.UseOutOfOrderExecution)
        {
            executor = new OutOfOrderBatchExecutor<ClassificationResult<bool>>(tokenizer.Tokenizer, executor);
        }

        var pipeline = new TextClassificationPipeline<bool>(tokenizer, modelExecutor, new TextClassificationOptions<bool>([false, true]), executor);
        return new SentimentInference(pipeline);
    }
}