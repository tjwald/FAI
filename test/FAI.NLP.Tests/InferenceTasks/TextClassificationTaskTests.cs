using System.Numerics.Tensors;
using FAI.Core.Abstractions;
using FAI.Core.Configurations.InferenceTasks;
using FAI.Core.ResultTypes;
using FAI.NLP.InferenceTasks.TextClassification;
using FAI.NLP.Tests.Mocks;
using FAI.NLP.Tokenization;

namespace FAI.NLP.Tests.InferenceTasks;

public class TextClassificationTaskTests
{
    [Fact]
    public async Task TextClassification_E2E_Mapping_Works()
    {
        // Arrange
        var tokenizer = DummyTokenizerFactory.Create();
        var modelExecutor = Substitute.For<IModelExecutor<long, float>>();

        var options = new ClassificationOptions<string>
        {
            Choices = ["Negative", "Positive"]
        };

        var task = new TextClassification<string>(tokenizer, modelExecutor, options);

        TokenizedText[] inputs = [new("hello")];
        var outputs = new ClassificationResult<string, float>[1];

        // Mock model output: 2 labels, [0.1f, 0.9f] -> Positive
        // The signature we use in ClassificationTask is usually RunAsync(Tensor<long>[] inputs, Action<ReadOnlyTensorSpan<float>, int> callback)
        // Wait, let's check ClassificationTask.cs if possible or assume standard.
        // Actually IModelExecutor has Task RunAsync(Tensor<TInput>[] inputs, Action<ReadOnlyTensorSpan<TOutput>, int> postProcess);

        modelExecutor.RunAsync(Arg.Any<Tensor<long>[]>(), Arg.Any<Action<ReadOnlyTensorSpan<float>, int>>())
            .Returns(x =>
            {
                var callback = x.ArgAt<Action<ReadOnlyTensorSpan<float>, int>>(1);
                var logits = Tensor.CreateFromShape<float>([1, 2]);
                logits[0, 0] = 0.1f;
                logits[0, 1] = 0.9f;

                // ReadOnlyTensorSpan is a ref struct, we might need a internal way to create it or just use Arg.Invoke if NSubstitute supports it for ref structs (it usually doesn't in that way)
                // However, we can use Arg.Do to capture the callback and then invoke it if we can construct the ref struct.
                callback(logits, 0);
                return Task.CompletedTask;
            });

        // Act
        // Use ProcessBatch as it is an IInferenceSteps
        await task.ProcessBatch(inputs, outputs);

        // Assert
        Assert.Equal("Positive", outputs[0].Choice);
        Assert.True(outputs[0].Score > 0.5f);
    }
}
