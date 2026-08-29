using System.Numerics.Tensors;
using FAI.Core.Configurations.InferenceTasks;
using FAI.Core.Pipelines;
using FAI.Core.ResultTypes;
using FAI.NLP.InferenceTasks.TextClassification;
using FAI.NLP.Tests.Mocks;
using FAI.NLP.Tokenization;

namespace FAI.NLP.Tests.InferenceTasks;

public class TextClassificationPipelinesTests
{
    [Fact]
    public async Task ClassificationPipelines_ComposeWithTensorOutputScope()
    {
        var tokenizer = DummyTokenizerFactory.Create();
        var options = new ClassificationOptions<string>(["Negative", "Positive"]);
        var encodingPipeline = new TextTensorization(tokenizer);
        var decodingPipeline = new ClassificationDecoding<string>(options);
        ReadOnlyMemory<TokenizedText> inputs = new TokenizedText[]
        {
            new("hello", tokenizer.Tokenize("hello").ToArray()),
            new("world", tokenizer.Tokenize("world").ToArray()),
        };
        Tensor<long>[] encoded = await encodingPipeline.ExecuteAsync(inputs, TestContext.Current.CancellationToken);
        Tensor<float> logits = Tensor.Create([0.1f, 0.9f, 0.8f, 0.2f], [2, 2]);
        using var outputs = new TestTensorOutputs(logits);
        Memory<ClassificationResult<string, float>> results =
            await decodingPipeline.ExecuteAsync(outputs, TestContext.Current.CancellationToken);

        Assert.Equal(2, encoded.Length);
        Assert.Equal(["Positive", "Negative"], results.ToArray().Select(result => result.Choice));
        Assert.All(results.ToArray(), result => Assert.True(result.Score > 0.5f));
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
