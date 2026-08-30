using System.Collections.Concurrent;
using System.Numerics.Tensors;
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

public sealed class EmbeddingPoolingPipeline : IPreallocatingPipeline<EmbeddingModelOutputs, Tensor<float>>
{
    public const int EmbeddingDimensions = 384;

    public bool TryAllocateOutput(EmbeddingModelOutputs input, out Tensor<float> output)
    {
        if (input.ModelOutputs.Count == 0)
        {
            output = null!;
            return false;
        }

        output = Tensor.CreateFromShape<float>([input.ModelOutputs.GetOutput(0).Lengths[0], EmbeddingDimensions]);
        return true;
    }

    public async ValueTask<Tensor<float>> ExecuteAsync(
        EmbeddingModelOutputs input,
        CancellationToken cancellationToken = default)
    {
        if (!TryAllocateOutput(input, out Tensor<float> output))
        {
            throw new InvalidOperationException("The embedding model did not produce an output tensor.");
        }

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

        Parallel.ForEach(
            Partitioner.Create(0, batchSize),
            new ParallelOptions { CancellationToken = cancellationToken },
            range =>
        {
            (int start, int end) = range;
            ReadOnlyTensorSpan<float> localTokenEmbeddings = input.ModelOutputs.GetOutput(0);
            TensorDimensionSpan<float> destinationRows = output.GetDimensionSpan(0);
            ReadOnlyTensorDimensionSpan<float> tokenEmbeddingBatches = localTokenEmbeddings.GetDimensionSpan(0);
            TensorDimensionSpan<long> attentionMaskRows = attentionMask.GetDimensionSpan(0);
            for (int batchIndex = start; batchIndex < end; batchIndex++)
            {
                MeanPoolAndNormalize(
                    tokenEmbeddingBatches[batchIndex],
                    attentionMaskRows[batchIndex],
                    destinationRows[batchIndex]);
            }
        });

        return ValueTask.CompletedTask;
    }

    private static void MeanPoolAndNormalize(
        scoped in ReadOnlyTensorSpan<float> tokenEmbeddings,
        scoped in ReadOnlyTensorSpan<long> attentionMask,
        scoped in TensorSpan<float> embedding)
    {
        bool hasIncludedTokens = false;
        ReadOnlyTensorDimensionSpan<float> tokenRows = tokenEmbeddings.GetDimensionSpan(0);
        int tokenCount = checked((int)tokenEmbeddings.Lengths[0]);

        for (int tokenIndex = 0; tokenIndex < tokenCount; tokenIndex++)
        {
            if (attentionMask[tokenIndex] == 0)
            {
                continue;
            }

            hasIncludedTokens = true;
            Tensor.Add(tokenRows[tokenIndex], embedding, embedding);
        }

        ReadOnlyTensorSpan<float> embeddingValues = embedding;
        float norm = Tensor.Norm(embeddingValues);
        if (!hasIncludedTokens || norm == 0)
        {
            return;
        }

        Tensor.Divide(embedding, norm, embedding);
    }
}
