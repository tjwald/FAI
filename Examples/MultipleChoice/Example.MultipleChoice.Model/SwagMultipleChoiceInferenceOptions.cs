using FAI.NLP.Configuration;

namespace Example.MultipleChoice.Model;

public record SwagMultipleChoiceInferenceOptions(
    string ModelDir,
    PretrainedTokenizerOptions TokenizerOptions,
    bool UseGpu = true)
{
    public static SwagMultipleChoiceInferenceOptions DefaultConfig
    {
        get
        {
            string modelDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "MultipleChoiceModelResources");
            return new SwagMultipleChoiceInferenceOptions(
                ModelDir: modelDir,
                TokenizerOptions: new PretrainedTokenizerOptions()
            );
        }
    }
}
