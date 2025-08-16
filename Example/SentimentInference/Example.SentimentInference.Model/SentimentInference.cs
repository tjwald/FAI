using FAI.Core.Abstractions;
using FAI.Core.ResultTypes;
using FAI.NLP.Tokenization;

namespace Example.SentimentInference.Model;

public sealed class SentimentInference : IInference<string, bool>
{
    private readonly IPipeline<TokenizedText, ClassificationResult<bool, float>> _pipeline;

    public SentimentInference(IPipeline<TokenizedText, ClassificationResult<bool, float>> pipeline)
    {
        _pipeline = pipeline;
    }

    public async Task<bool> Predict(string input)
    {
        ClassificationResult<bool, float> classificationResult = await _pipeline.Predict(input);
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

        ClassificationResult<bool, float>[] classificationResults = await _pipeline.BatchPredict(textInputs);
        return classificationResults.Select(x => x.Choice).ToArray();
    }
}