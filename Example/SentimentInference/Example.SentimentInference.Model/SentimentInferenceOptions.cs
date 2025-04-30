using ML.NLP.Configuration;
using ML.Onnx.Configuration;

namespace Example.SentimentInference.Model;

public record SentimentInferenceOptions(
    string ModelDir,
    PretrainedTokenizerOptions TokenizerOptions,
    ModelExecutorType ModelExecutorType,
    bool UseGpu = true)
{
    public static SentimentInferenceOptions DefaultConfig
    {
        get
        {
            var modelDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "ClassificationModelResources");
            return new SentimentInferenceOptions(
                ModelDir: modelDir,
                TokenizerOptions: new PretrainedTokenizerOptions(),
                ModelExecutorType: ModelExecutorType.Simple
            );
        }
    }
}