using FAI.NLP.PipelineBatchExecutors;

namespace FAI.IntegrationTests;

public class TextClassificationIntegrationTests
{
    [Fact]
    public async Task FullPipeline_ShouldClassifyText()
    {
        // Arrange
        var services = new ServiceCollection();

        var options = new ClassificationOptions<bool>([false, true]);
        services.AddSingleton(options);

        services.AddPipeline<TokenizedText, ClassificationResult<bool, float>>()
                .Use<TokenizerBatchExecutor<TokenizedText, ClassificationResult<bool, float>>>();

        var tokenizer = DummyTokenizerFactory.Create();
        services.AddSingleton<PretrainedTokenizer>(tokenizer);

        // Mock model: always returns high probability for 'true' (index 1)
        services.AddSingleton<IModelExecutor<long, float>>(new LogicalMockModelExecutor([[0.1f, 0.9f]]));

        services.AddSingleton<IInferenceSteps<TokenizedText, ClassificationResult<bool, float>>, TextClassification<bool>>();

        var provider = services.BuildServiceProvider();
        var pipeline = provider.GetRequiredService<IPipeline<TokenizedText, ClassificationResult<bool, float>>>();

        // Act
        var input = new TokenizedText("hello");
        var results = await pipeline.BatchPredict(new[] { input });

        // Assert
        results.Should().HaveCount(1);
        results[0].Choice.Should().BeTrue();
    }
}
