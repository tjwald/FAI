using System.Numerics.Tensors;
using FAI.Core.Abstractions;
using FAI.Core.Pipelines;

namespace Example.TextEmbedding.Model;

public sealed class TextEmbeddingInference : IBatchInference<string, Tensor<float>>
{
    private readonly IPipeline<ReadOnlyMemory<string>, Tensor<float>> _pipeline;
    private readonly TextEmbeddingOptions? _options;
    private int? _embeddingDimensions;

    public TextEmbeddingInference(
        IPipeline<ReadOnlyMemory<string>, Tensor<float>> pipeline,
        TextEmbeddingOptions? options = null)
    {
        _pipeline = pipeline;
        _options = options;
        _embeddingDimensions = options?.EmbeddingDimensions;
    }

    public Task<Tensor<float>> Predict(string input)
        => BatchPredict((string[])[input]);

    public async Task<Tensor<float>> BatchPredict(ReadOnlyMemory<string> input)
    {
        if (input.IsEmpty)
        {
            throw new ArgumentException("Cannot embed an empty text batch.", nameof(input));
        }

        if (_pipeline is IDestinationPipeline<ReadOnlyMemory<string>, Tensor<float>> destinationPipeline)
        {
            if (_embeddingDimensions is int dimensions)
            {
                Tensor<float> output = Tensor.CreateFromShape<float>([input.Length, dimensions]);
                await destinationPipeline.ExecuteAsync(input, output);
                return output;
            }

            Tensor<float> result = await _pipeline.ExecuteAsync(input);
            if (result.Lengths.Length >= 2)
            {
                _embeddingDimensions = checked((int)result.Lengths[1]);
            }

            return result;
        }

        return await _pipeline.ExecuteAsync(input);
    }
}
