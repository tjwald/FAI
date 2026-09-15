using System.Numerics.Tensors;
using FAI.Core;
using FAI.Core.Pipelines;
using FAI.NLP.Configuration;
using FAI.NLP.InferenceTasks.TextEmbedding;
using FAI.NLP.Tests.Mocks;
using FAI.NLP.Tokenization;

namespace FAI.NLP.Tests.InferenceTasks;

public class TextEmbeddingTaskTests
{
    [Fact]
    public async Task MeanPooling_ComputesNormalizedAverage()
    {
        var options = new TextEmbeddingOptions(PoolingStrategy.Mean, Normalize: true, EmbeddingDimensions: 2);
        var decoder = new TextEmbeddingDecoding(options);

        // Batch of 2, seq_len = 3, dim = 2
        // Item 0 has 2 tokens (index 2 is padding)
        // Item 1 has 1 token (index 1, 2 are padding)
        TokenizedText[] inputs =
        [
            new("two tokens", (int[])[10, 20]),
            new("one token", (int[])[30])
        ];

        // Tokens tensor: [2, 3, 2]
        // Item 0: [ [1, 0], [0, 1], [99, 99] ] -> sum = [1, 1], norm = sqrt(2), normalized = [1/sqrt(2), 1/sqrt(2)]
        // Item 1: [ [0, 3], [88, 88], [99, 99] ] -> sum = [0, 3], norm = 3, normalized = [0, 1]
        Tensor<float> tokenEmbeddings = Tensor.CreateFromShape<float>([2, 3, 2]);
        tokenEmbeddings[0, 0, 0] = 1f;
        tokenEmbeddings[0, 0, 1] = 0f;
        tokenEmbeddings[0, 1, 0] = 0f;
        tokenEmbeddings[0, 1, 1] = 1f;
        tokenEmbeddings[0, 2, 0] = 99f;
        tokenEmbeddings[0, 2, 1] = 99f;

        tokenEmbeddings[1, 0, 0] = 0f;
        tokenEmbeddings[1, 0, 1] = 3f;
        tokenEmbeddings[1, 1, 0] = 88f;
        tokenEmbeddings[1, 1, 1] = 88f;
        tokenEmbeddings[1, 2, 0] = 99f;
        tokenEmbeddings[1, 2, 1] = 99f;

        using var outputs = new TestTensorOutputs(tokenEmbeddings);
        Tensor<float> result = await decoder.ExecuteAsync((inputs, outputs), TestContext.Current.CancellationToken);

        float invSqrt2 = 1f / MathF.Sqrt(2f);
        Assert.Equal(2, result.Lengths[0]);
        Assert.Equal(2, result.Lengths[1]);
        Assert.Equal(invSqrt2, result[0, 0], precision: 4);
        Assert.Equal(invSqrt2, result[0, 1], precision: 4);
        Assert.Equal(0f, result[1, 0], precision: 4);
        Assert.Equal(1f, result[1, 1], precision: 4);
    }

    [Fact]
    public async Task ClsTokenPooling_ExtractsFirstTokenAndNormalizes()
    {
        var options = new TextEmbeddingOptions(PoolingStrategy.ClsToken, Normalize: true, EmbeddingDimensions: 2);
        var decoder = new TextEmbeddingDecoding(options);

        TokenizedText[] inputs =
        [
            new("text a", (int[])[1, 2]),
            new("text b", (int[])[3, 4])
        ];

        Tensor<float> tokenEmbeddings = Tensor.CreateFromShape<float>([2, 2, 2]);
        // Item 0: CLS = [3, 4] -> norm = 5, normalized = [0.6, 0.8]
        tokenEmbeddings[0, 0, 0] = 3f;
        tokenEmbeddings[0, 0, 1] = 4f;
        tokenEmbeddings[0, 1, 0] = 10f;
        tokenEmbeddings[0, 1, 1] = 10f;

        // Item 1: CLS = [0, -2] -> norm = 2, normalized = [0, -1]
        tokenEmbeddings[1, 0, 0] = 0f;
        tokenEmbeddings[1, 0, 1] = -2f;
        tokenEmbeddings[1, 1, 0] = 5f;
        tokenEmbeddings[1, 1, 1] = 5f;

        using var outputs = new TestTensorOutputs(tokenEmbeddings);
        Tensor<float> result = await decoder.ExecuteAsync((inputs, outputs), TestContext.Current.CancellationToken);

        Assert.Equal(0.6f, result[0, 0], precision: 4);
        Assert.Equal(0.8f, result[0, 1], precision: 4);
        Assert.Equal(0f, result[1, 0], precision: 4);
        Assert.Equal(-1f, result[1, 1], precision: 4);
    }

    [Fact]
    public async Task DestinationExecution_WritesDirectlyIntoSuppliedTensor()
    {
        var options = new TextEmbeddingOptions(PoolingStrategy.Mean, Normalize: false, EmbeddingDimensions: 2);
        var decoder = new TextEmbeddingDecoding(options);

        TokenizedText[] inputs = [new("single", (int[])[10, 20])];
        Tensor<float> tokenEmbeddings = Tensor.CreateFromShape<float>([1, 2, 2]);
        tokenEmbeddings[0, 0, 0] = 2f;
        tokenEmbeddings[0, 0, 1] = 4f;
        tokenEmbeddings[0, 1, 0] = 6f;
        tokenEmbeddings[0, 1, 1] = 8f;

        using var outputs = new TestTensorOutputs(tokenEmbeddings);
        Tensor<float> destination = Tensor.CreateFromShape<float>([1, 2]);
        await decoder.ExecuteAsync((inputs, outputs), destination, TestContext.Current.CancellationToken);

        // Average of [2, 4] and [6, 8] is [4, 6]
        Assert.Equal(4f, destination[0, 0], precision: 4);
        Assert.Equal(6f, destination[0, 1], precision: 4);
    }

    [Fact]
    public async Task TensorBatchInference_PredictAndBatchPredict_ProduceExpectedTensors()
    {
        var stubPipeline = new StubEmbeddingPipeline();
        var inference = new TensorBatchInference<string, float>(stubPipeline, secondaryDimension: 2);

        Tensor<float> single = await inference.Predict("hello");
        Assert.Equal([1, 2], single.Lengths.ToArray());

        Tensor<float> batch = await inference.BatchPredict((string[])["first", "second"]);
        Assert.Equal([2, 2], batch.Lengths.ToArray());

        Tensor<float> dest = Tensor.CreateFromShape<float>([2, 2]);
        await inference.BatchPredict((string[])["first", "second"], dest);
        Assert.Equal([2, 2], dest.Lengths.ToArray());
    }

    [Fact]
    public void PretrainedTokenizer_WithIncludeTokenTypeIds_GeneratesThreeTensors()
    {
        var options = new PretrainedTokenizerOptions(IncludeTokenTypeIds: true);
        var tokenizer = DummyTokenizerFactory.Create(options);
        List<int>[] inputs = [[1, 2], [3, 4]];

        var result = tokenizer.BatchTokensToTensors(inputs, 2);
        Tensor<long>[] tensorArray = result.ToArray();

        Assert.Equal(3, tensorArray.Length);
        Assert.NotNull(result.TokenTypeIds);
        Assert.Equal(result.Tokens.Lengths, result.TokenTypeIds.Lengths);
        Assert.All(result.TokenTypeIds.AsReadOnlyTensorSpan().AsSpan().ToArray(), val => Assert.Equal(0L, val));
    }

    private sealed class StubEmbeddingPipeline : IDestinationPipeline<ReadOnlyMemory<string>, Tensor<float>>
    {
        public ValueTask<Tensor<float>> ExecuteAsync(
            ReadOnlyMemory<string> input,
            CancellationToken cancellationToken = default)
        {
            Tensor<float> output = Tensor.CreateFromShape<float>([input.Length, 2]);
            return ValueTask.FromResult(output);
        }

        public ValueTask ExecuteAsync(
            ReadOnlyMemory<string> input,
            Tensor<float> destination,
            CancellationToken cancellationToken = default)
        {
            return ValueTask.CompletedTask;
        }
    }

    private sealed class TestTensorOutputs(Tensor<float> output) : TensorOutputs<float>
    {
        public override int Count => 1;

        public override ReadOnlyTensorSpan<float> GetOutput(int index)
            => index == 0 ? output.AsReadOnlyTensorSpan() : throw new ArgumentOutOfRangeException(nameof(index));

        public override void Dispose()
        {
        }
    }
}
