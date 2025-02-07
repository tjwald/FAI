using System.Numerics.Tensors;
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

public enum PipeLineExecutorType
{
    Serial,
    Parallel,
    Streamed,
}

public record SentimentInferenceOptions(
    string ModelDir,
    PretrainedTokenizerOptions TokenizerOptions,
    OnnxModelExecutorOptions OnnxModelExecutorOptions,
    int? MaxConcurrency,
    int BatchSize,
    PipeLineExecutorType PipeLineExecutorType,
    bool UseOutOfOrderExecution,
    ModelExecutorType ModelExecutorType,
    bool ParallelPreProcessing);

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
            _ => throw new NotImplementedException(nameof(options.ModelExecutorType)),
        };

        IPipelineBatchExecutor<string, ClassificationResult<bool>> executor = options.PipeLineExecutorType switch
        {
            PipeLineExecutorType.Serial => new SerialPipelineBatchExecutor<string, ClassificationResult<bool>>(maxBatchSize: options.BatchSize),
            PipeLineExecutorType.Parallel => new ParallelPipelineBatchExecutor<string, ClassificationResult<bool>>(options.BatchSize, options.MaxConcurrency),
            PipeLineExecutorType.Streamed => new StreamedBatchExecutor<string, ClassificationResult<bool>, BatchTokenizedResult, Tensor<float>[]>(options.BatchSize, options.MaxConcurrency, options.ParallelPreProcessing),
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