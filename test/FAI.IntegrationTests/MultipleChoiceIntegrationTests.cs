using FAI.NLP.Pipelines;

namespace FAI.IntegrationTests;

public class MultipleChoiceIntegrationTests
{
    [Fact]
    public async Task FullPipeline_ShouldHandleMultipleChoice()
    {
        var services = new ServiceCollection();
        services.AddSingleton(DummyTokenizerFactory.Create());
        services.AddSingleton(new TextMultipleChoiceOptions(MaxChoices: 2));
        services.AddSingleton<IPipeline<Tensor<long>[], TensorOutputs<float>>>(
            new LogicalMockModelPipeline([[0.9f, 0.1f]]));
        services
            .AddPipeline<ReadOnlyMemory<TextMultipleChoiceInput>>()
            .Then<ReadOnlyMemory<TokenizedTextMultipleChoiceInput>, TextMultipleChoiceTokenization>()
            .Then<Memory<ChoiceResult<TokenizedText>>, TextMultipleChoicePipeline>()
            .Build();

        await using ServiceProvider provider = services.BuildServiceProvider();
        var pipeline = provider.GetRequiredService<
            IPipeline<ReadOnlyMemory<TextMultipleChoiceInput>, Memory<ChoiceResult<TokenizedText>>>>();
        ReadOnlyMemory<TextMultipleChoiceInput> input = new TextMultipleChoiceInput[]
        {
            new("Question", [new("choice 1"), new("choice 2")]),
        };
        Memory<ChoiceResult<TokenizedText>> output =
            await pipeline.ExecuteAsync(input, TestContext.Current.CancellationToken);

        output.ToArray().Should().HaveCount(1);
        output.Span[0].ChoiceIndex.Should().Be(0);
    }
}
