using FAI.Core.Abstractions;
using FAI.Core.Pipelines;
using FAI.Core.ResultTypes;

namespace Example.SentimentInference.Model;

public sealed class SentimentInference :
    IInference<string, bool>,
    IInference<ReadOnlyMemory<string>, bool[]>
{
    private readonly IPipeline<ReadOnlyMemory<string>, Memory<ClassificationResult<bool, float>>> _pipeline;

    public SentimentInference(
        IPipeline<ReadOnlyMemory<string>, Memory<ClassificationResult<bool, float>>> pipeline)
    {
        _pipeline = pipeline;
    }

    public async Task<bool> Predict(string input)
    {
        bool[] output = await Predict((string[])[input]);
        return output[0];
    }

    public async Task<bool[]> Predict(ReadOnlyMemory<string> input)
    {
        var output = new bool[input.Length];
        await Predict(input, output);
        return output;
    }

    public async Task Predict(ReadOnlyMemory<string> input, bool[] destination)
    {
        if (input.Length != destination.Length)
        {
            throw new ArgumentException("Input and output batch sizes must match.", nameof(destination));
        }

        Memory<ClassificationResult<bool, float>> classificationResults = await _pipeline.ExecuteAsync(input);

        for (int index = 0; index < classificationResults.Length; index++)
        {
            destination[index] = classificationResults.Span[index].Choice;
        }
    }
}
