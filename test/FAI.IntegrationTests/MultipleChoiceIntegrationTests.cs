using FAI.NLP.PipelineBatchExecutors;

namespace FAI.IntegrationTests;

public class MultipleChoiceIntegrationTests
{
    [Fact]
    public async Task FullPipeline_ShouldHandleMultipleChoice()
    {
        // Arrange
        var services = new ServiceCollection();

        services.AddPipeline<TextMultipleChoiceInput, ChoiceResult<TokenizedText>>()
                .Use<TokenizerBatchExecutor<TextMultipleChoiceInput, ChoiceResult<TokenizedText>>>();

        var tokenizer = DummyTokenizerFactory.Create();
        services.AddSingleton<PretrainedTokenizer>(tokenizer);
        services.AddSingleton(new TextMultipleChoiceOptions { MaxChoices = 2 });

        // Mock model: always returns [0.9, 0.1] logits
        services.AddSingleton<IModelExecutor<long, float>>(new LogicalMockModelExecutor([[0.9f, 0.1f]]));

        services.AddSingleton<IInferenceSteps<TextMultipleChoiceInput, ChoiceResult<TokenizedText>>, TextMultipleChoiceTask>();

        var provider = services.BuildServiceProvider();
        var pipeline = provider.GetRequiredService<IPipeline<TextMultipleChoiceInput, ChoiceResult<TokenizedText>>>();

        var input = new TextMultipleChoiceInput(
            "Question",
            [new TokenizedText("choice 1"), new TokenizedText("choice 2")]
        );

        // Act
        var results = await pipeline.BatchPredict(new[] { input });

        // Assert
        results.Should().HaveCount(1);
        results[0].ChoiceIndex.Should().Be(0); // Chosen first choice
    }
}
