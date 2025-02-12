using ML.Infra.Abstractions;
using ML.Infra.ResultTypes;
using ML.Infra.Tokenization;

namespace Example.SentimentInference.Model;

public sealed class SentimentInference: IInference<string, bool>
{
    private readonly IPipeline<string, ClassificationResult<bool>> _pipeline;

    public SentimentInference(IPipeline<string, ClassificationResult<bool>> pipeline)
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

public sealed class SentimentInferenceV2: IInference<string, bool>
{
    private readonly IPipeline<TokenizedText, ClassificationResult<bool>> _pipeline;

    public SentimentInferenceV2(IPipeline<TokenizedText, ClassificationResult<bool>> pipeline)
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
        var textInputs = new TokenizedText[input.Length];
        ReadOnlySpan<string> inputSpan = input.Span; 
        for (int i = 0; i < input.Length; i++)
        {
            textInputs[i] = inputSpan[i];
        }
        
        ClassificationResult<bool>[] classificationResults = await _pipeline.BatchPredict(textInputs);
        return classificationResults.Select(x => x.Choice).ToArray();
    }
}