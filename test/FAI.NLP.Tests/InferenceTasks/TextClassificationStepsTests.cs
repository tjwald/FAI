using System.Numerics.Tensors;
using FAI.Core.Configurations.InferenceTasks;
using FAI.Core.ResultTypes;
using FAI.Core.Steps;
using FAI.NLP.InferenceTasks.TextClassification;
using FAI.NLP.Tests.Mocks;
using FAI.NLP.Tokenization;

namespace FAI.NLP.Tests.InferenceTasks;

public class TextClassificationStepsTests
{
    [Fact]
    public async Task ClassificationSteps_ComposeIntoCallerOutput()
    {
        var tokenizer = DummyTokenizerFactory.Create();
        var options = new ClassificationOptions<string>(["Negative", "Positive"]);
        var encodingStep = new TextBatchEncodingStep(tokenizer);
        var decodingStep = new ClassificationDecodingStep<string>(options);
        ReadOnlyMemory<TokenizedText> inputs = new TokenizedText[] { new("hello"), new("world") };
        var results = new ClassificationResult<string, float>[2];

        using BatchLease<Tensor<long>[]> encoded = await encodingStep.ExecuteAsync(inputs, TestContext.Current.CancellationToken);
        Tensor<float> logits = Tensor.Create([0.1f, 0.9f, 0.8f, 0.2f], [2, 2]);
        decodingStep.Consume(logits.AsReadOnlyTensorSpan(), 0, results);

        Assert.Equal(["Positive", "Negative"], results.Select(result => result.Choice));
        Assert.All(results, result => Assert.True(result.Score > 0.5f));
    }
}
