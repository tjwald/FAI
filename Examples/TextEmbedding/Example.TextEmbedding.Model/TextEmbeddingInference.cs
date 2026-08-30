using System.Numerics.Tensors;
using FAI.Core.Abstractions;
using FAI.Core.Pipelines;

namespace Example.TextEmbedding.Model;

public sealed class TextEmbeddingInference : IBatchInference<string, Tensor<float>>
{
    private readonly IPipeline<ReadOnlyMemory<string>, Tensor<float>> _pipeline;

    public TextEmbeddingInference(IPipeline<ReadOnlyMemory<string>, Tensor<float>> pipeline)
    {
        _pipeline = pipeline;
    }

    public Task<Tensor<float>> Predict(string input)
        => BatchPredict(new[] { input });

    public async Task<Tensor<float>> BatchPredict(ReadOnlyMemory<string> input)
    {
        if (input.IsEmpty)
        {
            throw new ArgumentException("Cannot embed an empty text batch.", nameof(input));
        }

        return await _pipeline.ExecuteAsync(input);
    }
}
