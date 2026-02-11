using System.Numerics.Tensors;
using FAI.Core.Abstractions;
using FAI.Core.ResultTypes;
using FAI.NLP.Configuration;
using FAI.NLP.InferenceTasks.TextMultipleChoice;
using FAI.NLP.Tests.Mocks;
using FAI.NLP.Tokenization;

namespace FAI.NLP.Tests.InferenceTasks;

public class TextMultipleChoiceTaskTests
{
    [Fact]
    public async Task TextMultipleChoice_FlatteningAndInference_Works()
    {
        // Arrange
        var tokenizer = DummyTokenizerFactory.Create();
        var modelExecutor = Substitute.For<IModelExecutor<long, float>>();

        var options = new TextMultipleChoiceOptions
        {
            MaxChoices = 4,
            StoreLogits = true
        };

        var task = new TextMultipleChoiceTask(tokenizer, modelExecutor, options);

        // Input with 2 choices
        var inputs = new TextMultipleChoiceInput[]
        {
            new("context", [new("choice 1"), new("choice 2")])
        };
        var outputs = new ChoiceResult<TokenizedText>[1];

        modelExecutor.RunAsync(Arg.Any<Tensor<long>[]>(), Arg.Any<Action<ReadOnlyTensorSpan<float>, int>>())
            .Returns(x =>
            {
                var callback = x.ArgAt<Action<ReadOnlyTensorSpan<float>, int>>(1);
                var logits = Tensor.CreateFromShape<float>([1, 2]);
                logits[0, 0] = -1.0f;
                logits[0, 1] = 2.0f; // Higher score for second choice
                callback(logits, 0);
                return Task.CompletedTask;
            });

        // Act
        await task.ProcessBatch(inputs, outputs);

        // Assert
        Assert.Equal(1, outputs[0].ChoiceIndex);
        Assert.Equal("choice 2", outputs[0].Choice.Text);
    }
}
