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
    public static SentimentInferenceOptions DefaultConfig
    {
        get
        {
            var modelDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "ClassificationModelResources");
            return new SentimentInferenceOptions(
                ModelDir: modelDir,
                TokenizerOptions: new PretrainedTokenizerOptions(PaddingToken: 0),
                ModelExecutorOptions: DefaultOnnxModelExecutorOptions(modelDir),
                PipeBatchExecutorOptions: new MaxPaddedTokensBatchExecutorOptions(new StreamedPipelineExecutorOptions(100000, 4, false), 2048, 0.1),
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