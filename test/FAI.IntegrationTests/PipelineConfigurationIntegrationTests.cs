using FAI.Core.BatchSlicers;
using FAI.NLP.PipelineBatchExecutors;

namespace FAI.IntegrationTests;

public class PipelineConfigurationIntegrationTests
{
    [Fact]
    public async Task ComplexPipeline_WithBackgroundAndPartitioning_ShouldProcessBatches()
    {
        // Arrange
        var services = new ServiceCollection();

        services.AddPipeline<TokenizedText, ClassificationResult<bool, float>>()
            .Use<BackgroundPipelineBatchExecutor<TokenizedText, ClassificationResult<bool, float>>>()
            .Use<PartitionPipelineBatchExecutor<TokenizedText, ClassificationResult<bool, float>>>()
            .Use<TokenizerBatchExecutor<TokenizedText, ClassificationResult<bool, float>>>();

        // Setup dependencies
        var tokenizer = DummyTokenizerFactory.Create();
        services.AddSingleton<PretrainedTokenizer>(tokenizer);
        services.AddSingleton<IBatchSchedular<TokenizedText, ClassificationResult<bool, float>>, ParallelBatchSchedular<TokenizedText, ClassificationResult<bool, float>>>();
        services.AddSingleton<IBatchSlicer<TokenizedText>, FixedSizeBatchSlicer<TokenizedText>>();

        // Add options for executors
        services.AddSingleton(new ParallelBatchSchedularOptions(2));
        services.AddSingleton(new FixedSizeBatchSlicerOptions(5));
        services.AddSingleton(new BackgroundPipelineBatchExecutorOptions(2));

        var options = new ClassificationOptions<bool>([false, true]);
        services.AddSingleton(options);

        // Mock model: always returns high probability for 'true'
        services.AddSingleton<IModelExecutor<long, float>>(new LogicalMockModelExecutor([[0.1f, 0.9f]]));

        services.AddSingleton<IInferenceSteps<TokenizedText, ClassificationResult<bool, float>>, TextClassification<bool>>();

        var provider = services.BuildServiceProvider();
        var pipeline = provider.GetRequiredService<IPipeline<TokenizedText, ClassificationResult<bool, float>>>();

        // Act
        var inputs = Enumerable.Range(0, 10).Select(i => new TokenizedText($"test {i}")).ToArray();
        var results = await pipeline.BatchPredict(inputs);

        // Assert
        results.Should().HaveCount(10);
        results.All(r => r.Choice).Should().BeTrue();
    }
}
