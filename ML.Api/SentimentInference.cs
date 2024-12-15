using ML.Infra.Abstractions;
using ML.Infra.Pipelines;

namespace WebApplication1;

public class SentimentInference: IInference<string, bool>
{
    private readonly TextClassificationPipeline<bool> _pipeline;

    public SentimentInference(TextClassificationPipeline<bool> pipeline)
    {
        _pipeline = pipeline;
    }

    public async Task<bool> Predict(string input)
    {
        ClassificationResult<bool> classificationResult = await _pipeline.Predict(input);
        return classificationResult.Choice;
    }

    public async Task<bool[]> BatchPredict(ReadOnlyMemory<string> input)
    {
        ClassificationResult<bool>[] classificationResults = await _pipeline.BatchPredict(input);
        return classificationResults.Select(x => x.Choice).ToArray();
    }
}