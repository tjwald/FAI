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
        var output = new bool[input.Length];
        await BatchPredict(input, output);
        return output;
    }

    public async Task BatchPredict(ReadOnlyMemory<string> input, Memory<bool> output)
    {
        var classificationResults = new ClassificationResult<bool, float>[input.Length];
        var textInputs = new TokenizedText[input.Length];
        ReadOnlySpan<string> inputSpan = input.Span;
        for (int i = 0; i < input.Length; i++)
        {
            textInputs[i] = inputSpan[i];
        }

        await _pipeline.BatchPredict(textInputs, classificationResults);

        Span<bool> outputSpan = output.Span;
        for (int i = 0; i < classificationResults.Length; i++)
        {
            outputSpan[i] = classificationResults[i].Choice;
        }
    }
}
