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
    public static SwagMultipleChoiceInferenceOptions DefaultConfig
    {
        get
        {
            string modelDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "MultipleChoiceModelResources");
            return new SwagMultipleChoiceInferenceOptions(
                ModelDir: modelDir,
                TokenizerOptions: new PretrainedTokenizerOptions(PaddingToken: 0),
                ModelExecutorOptions: DefaultOnnxModelExecutorOptions(modelDir),
                PipeBatchExecutorOptions: new MaxPaddedTokensBatchExecutorOptions(new StreamedPipelineExecutorOptions(100000, 5, false), 8192, 0.1),
                ModelExecutorType: ModelExecutorType.Simple
            );
        }
    }

    private static OnnxModelExecutorOptions DefaultOnnxModelExecutorOptions(string modelDir, bool useGpu = true)
    {
        var executorOptions = new OnnxModelExecutorOptions();
        return executorOptions.ConfigureOnnxOptions(onnxOptions =>
        {
            onnxOptions.ConfigureSessionOptions(options =>
            {
                if (useGpu)
                {
                    options.AppendExecutionProvider_CUDA();
                    Console.WriteLine("Using GPU accelerator");
                }

                options.AppendExecutionProvider_CPU();
            });
            onnxOptions.ModelDir = modelDir;
        });
    }
}
