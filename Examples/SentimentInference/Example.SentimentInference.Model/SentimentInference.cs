using FAI.Core.Abstractions;
using FAI.Core.ResultTypes;
using FAI.Core.Steps;

namespace Example.SentimentInference.Model;

public sealed class SentimentInference : IInference<string, bool>
{
    private readonly IStep<ReadOnlyMemory<string>, Memory<ClassificationResult<bool, float>>> _pipeline;

    public SentimentInference(
        IStep<ReadOnlyMemory<string>, Memory<ClassificationResult<bool, float>>> pipeline)
    {
        _pipeline = pipeline;
    }

    public async Task<bool> Predict(string input)
    {
        bool[] output = await BatchPredict(new[] { input });
        return output[0];
    }

    public async Task<bool[]> BatchPredict(ReadOnlyMemory<string> input)
    {
        var output = new bool[input.Length];
        await BatchPredict(input, output);
        return output;
    }

    public async Task BatchPredict(ReadOnlyMemory<string> input, Memory<bool> output)
    {
        if (input.Length != output.Length)
        {
            throw new ArgumentException("Input and output batch sizes must match.", nameof(output));
        }

        Memory<ClassificationResult<bool, float>> classificationResults = await _pipeline.ExecuteAsync(input);

        for (int index = 0; index < classificationResults.Length; index++)
        {
            output.Span[index] = classificationResults.Span[index].Choice;
        }
    }
}
