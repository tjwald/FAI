using Microsoft.ML.OnnxRuntime;
using ML.Infra.Configurations.ModelExecutors;
using ML.Infra.Configurations.PipelineBatchExecutors;
using ML.NLP.Configuration;
using ML.Onnx.Configuration;
using ML.Onnx.Factories;

namespace Example.SentimentInference.Model;

public record SentimentInferenceOptions(
    string ModelDir,
    PretrainedTokenizerOptions TokenizerOptions,
    IModelExecutorConfig ModelExecutorOptions,
    IPipelineBatchExecutorOptions PipeBatchExecutorOptions,
    ModelExecutorType ModelExecutorType)
{
    public static readonly SentimentInferenceOptions DefaultConfig = new(
        ModelDir: Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "ClassificationModelResources"),
        TokenizerOptions: new PretrainedTokenizerOptions(PaddingToken: 0),
        ModelExecutorOptions: new OnnxModelExecutorOptions(UseGpu: true, ExecutionMode: ExecutionMode.ORT_SEQUENTIAL, MaxThreads: null),
        PipeBatchExecutorOptions: new MaxPaddedTokensBatchExecutorOptions(new StreamedPipelineExecutorOptions(100000, 4, false), 2048, 0.1),
        ModelExecutorType: ModelExecutorType.Simple
    );
}