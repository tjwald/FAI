using FAI.NLP.Pipelines;

namespace FAI.IntegrationTests;

public class TextClassificationIntegrationTests
{
    [Fact]
    public async Task FullPipeline_ShouldClassifyText()
    {
        var services = new ServiceCollection();
        services.AddSingleton(new ClassificationOptions<bool>([false, true]));
        services.AddSingleton(DummyTokenizerFactory.Create());
        services.AddSingleton<IPipeline<BatchEncode, TensorOutputs<float>>>(
            new LogicalMockModelPipeline([[0.1f, 0.9f]]));
        services.AddSingleton<ClassificationDecoding<bool>>();
        services
            .AddPipeline<ReadOnlyMemory<string>>()
            .Then<ReadOnlyMemory<TokenizedText>, TextTokenization>()
            .Then<BatchEncode, TextTensorization>()
            .Then(sp =>
                sp.GetRequiredService<IPipeline<BatchEncode, TensorOutputs<float>>>())
            .Then<Memory<ClassificationResult<bool, float>>, ClassificationDecoding<bool>>()
            .Build();

        await using ServiceProvider provider = services.BuildServiceProvider();
        var pipeline = provider.GetRequiredService<
            IPipeline<ReadOnlyMemory<string>, Memory<ClassificationResult<bool, float>>>>();
        string[] input = ["hello"];
        Memory<ClassificationResult<bool, float>> output =
            await pipeline.ExecuteAsync(input, TestContext.Current.CancellationToken);

        output.ToArray().Should().HaveCount(1);
        output.Span[0].Choice.Should().BeTrue();
    }
}
