using System.Numerics.Tensors;
using FAI.Core;
using FAI.Core.Pipelines;

namespace Example.TextEmbedding.Model;

public sealed class EmbeddingPoolingPipeline : IPreallocatingPipeline<Tensor<long>[], Tensor<float>>
{
    public const int EmbeddingDimensions = 384;

    private readonly IPipeline<Tensor<long>[], TensorOutputs<float>> _modelPipeline;

    public EmbeddingPoolingPipeline(IPipeline<Tensor<long>[], TensorOutputs<float>> modelPipeline)
    {
        _modelPipeline = modelPipeline;
    }

    public bool TryAllocateOutput(Tensor<long>[] input, out Tensor<float> output)
    {
        if (input.Length < 2)
        {
            output = null!;
            return false;
        }

        output = Tensor.CreateFromShape<float>([input[0].Lengths[0], EmbeddingDimensions]);
        return true;
    }

    public async ValueTask<Tensor<float>> ExecuteAsync(
        Tensor<long>[] input,
        CancellationToken cancellationToken = default)
    {
        if (!TryAllocateOutput(input, out Tensor<float> output))
        {
            throw new ArgumentException("Embedding inference requires token and attention-mask tensors.", nameof(input));
        }

        await ExecuteAsync(input, output, cancellationToken);
        return output;
    }

    public async ValueTask ExecuteAsync(
        Tensor<long>[] input,
        Tensor<float> output,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (input.Length < 2)
        {
            throw new ArgumentException("Embedding inference requires token and attention-mask tensors.", nameof(input));
        }

        Tensor<long>[] modelInput = AddTokenTypeIds(input);
        using TensorOutputs<float> modelOutputs = await _modelPipeline.ExecuteAsync(modelInput, cancellationToken);
        if (modelOutputs.Count == 0)
        {
            throw new InvalidOperationException("The embedding model did not produce an output tensor.");
        }

        ReadOnlyTensorSpan<float> tokenEmbeddings = modelOutputs.GetOutput(0);
        if (tokenEmbeddings.Rank != 3)
        {
            throw new InvalidOperationException($"Expected a rank-3 token embedding tensor, but received rank {tokenEmbeddings.Rank}.");
        }

        int batchSize = checked((int)tokenEmbeddings.Lengths[0]);
        int tokenCount = checked((int)tokenEmbeddings.Lengths[1]);
        int dimensions = checked((int)tokenEmbeddings.Lengths[2]);
        Tensor<long> attentionMask = input[1];

        if (dimensions != EmbeddingDimensions)
        {
            throw new InvalidOperationException($"Expected {EmbeddingDimensions} embedding dimensions, but the model produced {dimensions}.");
        }

        if (output.Rank != 2 || output.Lengths[0] != batchSize || output.Lengths[1] != dimensions ||
            attentionMask.Lengths[0] != batchSize || attentionMask.Lengths[1] != tokenCount)
        {
            throw new ArgumentException("The output buffer and attention mask must match the model output shape.", nameof(output));
        }

        TensorDimensionSpan<float> destinationRows = output.GetDimensionSpan(0);
        ReadOnlySpan<float> tokenEmbeddingValues = tokenEmbeddings.AsSpan();
        for (int batchIndex = 0; batchIndex < batchSize; batchIndex++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            MeanPoolAndNormalize(
                tokenEmbeddingValues,
                attentionMask,
                batchIndex,
                tokenCount,
                destinationRows[batchIndex].AsSpan());
        }
    }

    private static Tensor<long>[] AddTokenTypeIds(Tensor<long>[] input)
    {
        if (input.Length != 2)
        {
            throw new ArgumentException("MiniLM embedding inference requires token and attention-mask tensors.", nameof(input));
        }

        Tensor<long> tokenTypeIds = Tensor.CreateFromShape<long>(input[0].Lengths);
        return [input[0], input[1], tokenTypeIds];
    }

    private static void MeanPoolAndNormalize(
        ReadOnlySpan<float> tokenEmbeddings,
        Tensor<long> attentionMask,
        int batchIndex,
        int tokenCount,
        Span<float> embedding)
    {
        bool hasIncludedTokens = false;

        for (int tokenIndex = 0; tokenIndex < tokenCount; tokenIndex++)
        {
            if (attentionMask[batchIndex, tokenIndex] == 0)
            {
                continue;
            }

            hasIncludedTokens = true;
            int tokenOffset = ((batchIndex * tokenCount) + tokenIndex) * embedding.Length;
            TensorPrimitives.Add(tokenEmbeddings.Slice(tokenOffset, embedding.Length), embedding, embedding);
        }

        if (!hasIncludedTokens)
        {
            return;
        }

        float squaredNorm = 0;
        for (int dimension = 0; dimension < embedding.Length; dimension++)
        {
            squaredNorm += embedding[dimension] * embedding[dimension];
        }

        float norm = MathF.Sqrt(squaredNorm);
        if (norm == 0)
        {
            return;
        }

        for (int dimension = 0; dimension < embedding.Length; dimension++)
        {
            embedding[dimension] /= norm;
        }
    }
}
