using System.Numerics.Tensors;
using FAI.Core;
using FAI.Core.Pipelines;
using FAI.NLP.Configuration;
using FAI.NLP.Tokenization;

namespace FAI.NLP.InferenceTasks.TextEmbedding;

public sealed class TextEmbeddingDecoding :
    IDestinationPipeline<(ReadOnlyMemory<TokenizedText> Input, TensorOutputs<float> ModelOutputs), Tensor<float>>
{
    public const int DefaultEmbeddingDimensions = 384;
    private readonly TextEmbeddingOptions _options;

    public TextEmbeddingDecoding(TextEmbeddingOptions? options = null)
    {
        _options = options ?? new TextEmbeddingOptions();
    }

    public async ValueTask<Tensor<float>> ExecuteAsync(
        (ReadOnlyMemory<TokenizedText> Input, TensorOutputs<float> ModelOutputs) input,
        CancellationToken cancellationToken = default)
    {
        if (input.ModelOutputs.Count == 0)
        {
            throw new InvalidOperationException("The embedding model did not produce an output tensor.");
        }

        ReadOnlyTensorSpan<float> tokenEmbeddings = input.ModelOutputs.GetOutput(0);
        int dimensions = checked((int)tokenEmbeddings.Lengths[2]);
        Tensor<float> output = Tensor.CreateFromShape<float>([tokenEmbeddings.Lengths[0], dimensions]);
        await ExecuteAsync(input, output, cancellationToken);
        return output;
    }

    public ValueTask ExecuteAsync(
        (ReadOnlyMemory<TokenizedText> Input, TensorOutputs<float> ModelOutputs) input,
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
        ReadOnlyMemory<TokenizedText> tokenizedTexts = input.Input;

        if (_options.EmbeddingDimensions is int expectedDimensions && expectedDimensions != dimensions)
        {
            throw new InvalidOperationException($"Expected {expectedDimensions} embedding dimensions, but the model produced {dimensions}.");
        }

        if (output.Rank != 2 || output.Lengths[0] != batchSize || output.Lengths[1] != dimensions ||
            tokenizedTexts.Length != batchSize)
        {
            throw new ArgumentException("The output buffer and input tokens must match the model output shape.", nameof(output));
        }

        ReadOnlyTensorDimensionSpan<float> sentenceBatch = tokenEmbeddings.GetDimensionSpan(0);
        TensorDimensionSpan<float> outputRows = output.GetDimensionSpan(0);
        ReadOnlySpan<TokenizedText> tokensSpan = tokenizedTexts.Span;

        PoolingStrategy strategy = _options.PoolingStrategy;
        bool normalize = _options.Normalize;

        if (strategy == PoolingStrategy.ClsToken)
        {
            PoolClsToken(sentenceBatch, outputRows, normalize);
        }
        else
        {
            PoolMean(sentenceBatch, outputRows, tokensSpan, normalize);
        }

        return ValueTask.CompletedTask;
    }

    private static void PoolClsToken(
        ReadOnlyTensorDimensionSpan<float> sentenceBatch,
        TensorDimensionSpan<float> outputRows,
        bool normalize)
    {
        for (int batchIndex = 0; batchIndex < sentenceBatch.Length; batchIndex++)
        {
            ReadOnlyTensorSpan<float> sentence = sentenceBatch[batchIndex];
            Span<float> embedding = outputRows[batchIndex].AsSpan();

            ReadOnlyTensorSpan<float> clsToken = sentence.GetDimensionSpan(0)[0];
            clsToken.AsSpan().CopyTo(embedding);

            if (normalize)
            {
                float norm = TensorPrimitives.Norm(embedding);
                if (norm > 0)
                {
                    TensorPrimitives.Divide(embedding, norm, embedding);
                }
            }
        }
    }

    private static void PoolMean(
        ReadOnlyTensorDimensionSpan<float> sentenceBatch,
        TensorDimensionSpan<float> outputRows,
        ReadOnlySpan<TokenizedText> tokensSpan,
        bool normalize)
    {
        for (int batchIndex = 0; batchIndex < sentenceBatch.Length; batchIndex++)
        {
            ReadOnlyTensorSpan<float> sentence = sentenceBatch[batchIndex];
            Span<float> embedding = outputRows[batchIndex].AsSpan();
            int realTokenCount = Math.Min((int)sentence.Lengths[0], tokensSpan[batchIndex].TokenCount);

            embedding.Clear();

            ReadOnlyTensorDimensionSpan<float> tokenRows = sentence.GetDimensionSpan(0);
            for (int tokenIndex = 0; tokenIndex < realTokenCount; tokenIndex++)
            {
                ReadOnlyTensorSpan<float> tokenRow = tokenRows[tokenIndex];
                TensorPrimitives.Add(embedding, tokenRow.AsSpan(), embedding);
            }

            if (!normalize && realTokenCount > 0)
            {
                TensorPrimitives.Divide(embedding, realTokenCount, embedding);
            }
            else if (normalize && realTokenCount > 0)
            {
                float norm = TensorPrimitives.Norm(embedding);
                if (norm > 0)
                {
                    TensorPrimitives.Divide(embedding, norm, embedding);
                }
            }
        }
    }
}
