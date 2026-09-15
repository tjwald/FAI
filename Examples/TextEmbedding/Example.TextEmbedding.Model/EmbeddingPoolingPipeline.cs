using System.Numerics.Tensors;
using FAI.Core;
using FAI.Core.Pipelines;

namespace Example.TextEmbedding.Model;

public sealed class EmbeddingModelOutputs : IDisposable
{
    public EmbeddingModelOutputs(TensorOutputs<float> modelOutputs, Tensor<long> attentionMask)
    {
        ModelOutputs = modelOutputs;
        AttentionMask = attentionMask;
    }

    public TensorOutputs<float> ModelOutputs { get; }
    public Tensor<long> AttentionMask { get; }

    public void Dispose() => ModelOutputs.Dispose();
}

public sealed class EmbeddingModelPipeline : IPipeline<Tensor<long>[], EmbeddingModelOutputs>
{
    private readonly IPipeline<Tensor<long>[], TensorOutputs<float>> _modelPipeline;

    public EmbeddingModelPipeline(IPipeline<Tensor<long>[], TensorOutputs<float>> modelPipeline)
    {
        _modelPipeline = modelPipeline;
    }

    public async ValueTask<EmbeddingModelOutputs> ExecuteAsync(
        Tensor<long>[] input,
        CancellationToken cancellationToken = default)
    {
        if (input.Length != 2)
        {
            throw new ArgumentException("MiniLM embedding inference requires token and attention-mask tensors.", nameof(input));
        }

        Tensor<long> tokenTypeIds = Tensor.CreateFromShape<long>(input[0].Lengths);
        TensorOutputs<float> modelOutputs = await _modelPipeline.ExecuteAsync([input[0], input[1], tokenTypeIds], cancellationToken);
        return new EmbeddingModelOutputs(modelOutputs, input[1]);
    }
}

public sealed class EmbeddingPoolingPipeline : IDestinationPipeline<EmbeddingModelOutputs, Tensor<float>>
{
    public const int EmbeddingDimensions = 384;

    public async ValueTask<Tensor<float>> ExecuteAsync(
        EmbeddingModelOutputs input,
        CancellationToken cancellationToken = default)
    {
        if (input.ModelOutputs.Count == 0)
        {
            throw new InvalidOperationException("The embedding model did not produce an output tensor.");
        }

        Tensor<float> output = Tensor.CreateFromShape<float>([input.ModelOutputs.GetOutput(0).Lengths[0], EmbeddingDimensions]);
        await ExecuteAsync(input, output, cancellationToken);
        return output;
    }

    public ValueTask ExecuteAsync(
        EmbeddingModelOutputs input,
        Tensor<float> output,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (input.ModelOutputs.Count == 0)
        {
            throw new InvalidOperationException("The embedding model did not produce an output tensor.");
        }

        ReadOnlyTensorSpan<float> tokenEmbeddings = input.ModelOutputs.GetOutput(0);
        if (tokenEmbeddings.Rank != 3)
        {
            throw new InvalidOperationException($"Expected a rank-3 token embedding tensor, but received rank {tokenEmbeddings.Rank}.");
        }

        int batchSize = checked((int)tokenEmbeddings.Lengths[0]);
        int tokenCount = checked((int)tokenEmbeddings.Lengths[1]);
        int dimensions = checked((int)tokenEmbeddings.Lengths[2]);
        Tensor<long> attentionMask = input.AttentionMask;

        if (dimensions != EmbeddingDimensions)
        {
            throw new InvalidOperationException($"Expected {EmbeddingDimensions} embedding dimensions, but the model produced {dimensions}.");
        }

        if (output.Rank != 2 || output.Lengths[0] != batchSize || output.Lengths[1] != dimensions ||
            attentionMask.Lengths[0] != batchSize || attentionMask.Lengths[1] != tokenCount)
        {
            throw new ArgumentException("The output buffer and attention mask must match the model output shape.", nameof(output));
        }

        ReadOnlySpan<float> allTokens = tokenEmbeddings.AsSpan();
        Span<float> allOutput = output.AsTensorSpan().AsSpan();
        ReadOnlySpan<long> allMask = attentionMask.AsTensorSpan().AsSpan();

        for (int batchIndex = 0; batchIndex < batchSize; batchIndex++)
        {
            Span<float> embedding = allOutput.Slice(batchIndex * dimensions, dimensions);
            ReadOnlySpan<long> mask = allMask.Slice(batchIndex * tokenCount, tokenCount);
            int batchTokenOffset = batchIndex * tokenCount * dimensions;
            bool hasIncludedTokens = false;

            for (int tokenIndex = 0; tokenIndex < tokenCount; tokenIndex++)
            {
                if (mask[tokenIndex] == 0)
                {
                    continue;
                }

                hasIncludedTokens = true;
                ReadOnlySpan<float> tokenRow = allTokens.Slice(batchTokenOffset + tokenIndex * dimensions, dimensions);
                TensorPrimitives.Add(embedding, tokenRow, embedding);
            }

            float norm = TensorPrimitives.Norm(embedding);
            if (hasIncludedTokens && norm > 0)
            {
                TensorPrimitives.Divide(embedding, norm, embedding);
            }
        }

        return ValueTask.CompletedTask;
    }
}
