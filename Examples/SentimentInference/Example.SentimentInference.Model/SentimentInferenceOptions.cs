using FAI.NLP.Configuration;

namespace Example.SentimentInference.Model;

public record SentimentInferenceOptions(
    string ModelDir,
    PretrainedTokenizerOptions TokenizerOptions,
    bool UseGpu = true)
{
    public static SentimentInferenceOptions DefaultConfig
    {
        get
        {
            var modelDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "ClassificationModelResources");
            return new SentimentInferenceOptions(
                ModelDir: modelDir,
                TokenizerOptions: new PretrainedTokenizerOptions()
            );
        }
    }
}
