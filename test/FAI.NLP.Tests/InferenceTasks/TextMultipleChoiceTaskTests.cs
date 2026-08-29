using System.Numerics.Tensors;
using FAI.Core.ResultTypes;
using FAI.Core.Steps;
using FAI.NLP.Configuration;
using FAI.NLP.InferenceTasks.TextMultipleChoice;
using FAI.NLP.Steps;
using FAI.NLP.Tests.Mocks;
using FAI.NLP.Tokenization;

namespace FAI.NLP.Tests.InferenceTasks;

public class TextMultipleChoiceTaskTests
{
    [Fact]
    public async Task TextMultipleChoiceStep_ProcessesBatchIntoCallerOutput()
    {
        var tokenizer = DummyTokenizerFactory.Create();
        var modelStep = new StubMultipleChoiceModelStep();
        var options = new TextMultipleChoiceOptions(MaxChoices: 4, StoreLogits: true);
        var tokenizationStep = new TextMultipleChoiceTokenizationStep(tokenizer);
        var step = new TextMultipleChoiceStep(modelStep, options);
        ReadOnlyMemory<TextMultipleChoiceInput> inputs = new TextMultipleChoiceInput[]
        {
            new("context", [new("choice 1"), new("choice 2")]),
            new("context", [new("choice a"), new("choice b")]),
        };
        ReadOnlyMemory<TokenizedTextMultipleChoiceInput> tokenizedInputs =
            await tokenizationStep.ExecuteAsync(inputs, TestContext.Current.CancellationToken);
        var output = new ChoiceResult<TokenizedText>[2];

        await step.ExecuteAsync(tokenizedInputs, output, TestContext.Current.CancellationToken);

        Assert.Equal([1, 0], output.Select(result => result.ChoiceIndex));
        Assert.Equal(["choice 2", "choice a"], output.Select(result => result.Choice.Text));
        Assert.Equal(2, modelStep.BatchSize);
    }

    private sealed class StubMultipleChoiceModelStep : IStep<Tensor<long>[], TensorOutputs<float>>
    {
        public int BatchSize { get; private set; }

        public ValueTask<TensorOutputs<float>> ExecuteAsync(
            Tensor<long>[] input,
            CancellationToken cancellationToken = default)
        {
            BatchSize = checked((int)input[0].Lengths[0]);
            int choices = checked((int)input[0].Lengths[1]);
            Tensor<float> logits = Tensor.CreateFromShape<float>([BatchSize, choices]);
            logits[0, 0] = -1.0f;
            logits[0, 1] = 2.0f;
            logits[1, 0] = 3.0f;
            logits[1, 1] = -2.0f;
            return ValueTask.FromResult<TensorOutputs<float>>(new TestTensorOutputs(logits));
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
