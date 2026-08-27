namespace FAI.IntegrationTests;

public class MultipleChoiceIntegrationTests
{
    [Fact]
    public async Task FullPipeline_ShouldHandleMultipleChoice()
    {
        var services = new ServiceCollection();
        services.AddSingleton(DummyTokenizerFactory.Create());
        services.AddSingleton(new TextMultipleChoiceOptions(MaxChoices: 2));
        services.AddSingleton<IBorrowedTensorProducer<Tensor<long>[], float>>(
            new LogicalMockModelStep([[0.9f, 0.1f]]));
        services
            .AddPipeline<ReadOnlyMemory<TextMultipleChoiceInput>>()
            .Then<Memory<ChoiceResult<TokenizedText>>, TextMultipleChoiceStep>(stage => stage.UseTokenizingStep())
            .Build();

        await using ServiceProvider provider = services.BuildServiceProvider();
        var pipeline = provider.GetRequiredService<
            IStep<ReadOnlyMemory<TextMultipleChoiceInput>, Memory<ChoiceResult<TokenizedText>>>>();
        ReadOnlyMemory<TextMultipleChoiceInput> input = new TextMultipleChoiceInput[]
        {
            new("Question", [new("choice 1"), new("choice 2")]),
        };
        var output = new ChoiceResult<TokenizedText>[1];

        await pipeline.ExecuteAsync(input, output, TestContext.Current.CancellationToken);

        output.Should().HaveCount(1);
        output[0].ChoiceIndex.Should().Be(0);
    }
}
