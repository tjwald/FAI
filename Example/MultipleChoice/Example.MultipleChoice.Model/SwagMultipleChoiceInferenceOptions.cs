using Microsoft.ML.OnnxRuntime;
using ML.Infra.Configurations.ModelExecutors;
using ML.Infra.Configurations.PipelineBatchExecutors;
using ML.NLP.Configuration;
using ML.Onnx.Configuration;
using ML.Onnx.Factories;

namespace Example.MultipleChoice.Model;

public record SwagMultipleChoiceInferenceOptions(
    string ModelDir,
    PretrainedTokenizerOptions TokenizerOptions,
    IModelExecutorConfig ModelExecutorOptions,
    IPipelineBatchExecutorOptions PipeBatchExecutorOptions,
    ModelExecutorType ModelExecutorType)
{
    public static readonly SwagMultipleChoiceInferenceOptions DefaultConfig = new(
        ModelDir: Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "MultipleChoiceModelResources"),
        TokenizerOptions: new PretrainedTokenizerOptions(PaddingToken: 0),
        ModelExecutorOptions: new OnnxModelExecutorOptions(UseGpu: true, ExecutionMode: ExecutionMode.ORT_SEQUENTIAL, MaxThreads: null),
        PipeBatchExecutorOptions: new MaxPaddedTokensBatchExecutorOptions(new StreamedPipelineExecutorOptions(100000, 5, false), 8192, 0.1),
        ModelExecutorType: ModelExecutorType.Simple
    );
}
