using FAI.NLP.Configuration;
using FAI.Onnx.Configuration;

namespace Example.MultipleChoice.Model;

public record SwagMultipleChoiceInferenceOptions(
    string ModelDir,
    PretrainedTokenizerOptions TokenizerOptions,
    ModelExecutorType ModelExecutorType,
    bool UseGpu = true)
{
    public static SwagMultipleChoiceInferenceOptions DefaultConfig
    {
        get
        {
            string modelDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "MultipleChoiceModelResources");
            return new SwagMultipleChoiceInferenceOptions(
                ModelDir: modelDir,
                TokenizerOptions: new PretrainedTokenizerOptions(),
                ModelExecutorType: ModelExecutorType.Simple
            );
        }
    }
}
