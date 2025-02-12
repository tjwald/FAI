using Microsoft.ML.OnnxRuntime;
using ML.Infra.Configurations.ModelExecutors;
using ML.Infra.Configurations.PipelineBatchExecutors;
using ML.Infra.Factories;
using ML.Infra.Tokenization;

namespace Example.SentimentInference.Model;

public record SentimentInferenceOptions(
    string ModelDir,
    PretrainedTokenizerOptions TokenizerOptions,
    IModelExecutorConfig OnnxModelExecutorOptions,
    int? MaxConcurrency,
    int BatchSize,
    PipelineExecutorType PipelineExecutorType,
    bool UseTokenSortingExecution,
    ModelExecutorType ModelExecutorType,
    bool ParallelPreProcessing)
{
    public static readonly SentimentInferenceOptions DefaultConfig = new(
        ModelDir: Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "ClassificationModelResources"),
        TokenizerOptions: new PretrainedTokenizerOptions(PaddingToken: 0),
        new OnnxModelExecutorOptions(UseGpu: true, ExecutionMode: ExecutionMode.ORT_SEQUENTIAL, MaxThreads: null),
        MaxConcurrency: 4,
        BatchSize: 400,
        ModelExecutorType: ModelExecutorType.Simple,
        UseTokenSortingExecution: true,
        PipelineExecutorType: PipelineExecutorType.Streamed,
        ParallelPreProcessing: true
    );
}