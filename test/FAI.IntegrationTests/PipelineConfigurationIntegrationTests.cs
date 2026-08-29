using FAI.NLP.Pipelines;

namespace FAI.IntegrationTests;

public class PipelineConfigurationIntegrationTests
{
    [Fact]
    public async Task FinitePolicies_ProcessPartitionedBatchAndRestoreOrder()
    {
        var services = new ServiceCollection();
        services.AddSingleton(DummyTokenizerFactory.Create());
        services.AddSingleton(new ClassificationOptions<bool>([false, true]));
        services.AddSingleton(new TokenCountOrderingOptions(Ascending: true));
        services.AddSingleton(new MaxPaddedTokensPartitionerOptions(MaxPaddedTokenRatio: 1, MaxTokenCount: 20));
        services.AddSingleton<IPartitionScheduler>(
            new ParallelPartitionScheduler(new ParallelPartitionSchedulerOptions(MaxConcurrency: 2)));
        services.AddSingleton<IPipeline<Tensor<long>[], TensorOutputs<float>>>(
            new LogicalMockModelPipeline([[0.1f, 0.9f]]));
        services.AddSingleton<ClassificationDecoding<bool>>();
        services
            .AddPipeline<ReadOnlyMemory<string>>()
            .Then<ReadOnlyMemory<TokenizedText>, TextTokenization>()
            .UseTokenCountOrdering()
            .UseMaxPaddedTokensPartitioning()
            .Then<Tensor<long>[], TextTensorization>()
            .Then(sp =>
                sp.GetRequiredService<IPipeline<Tensor<long>[], TensorOutputs<float>>>())
            .Then<Memory<ClassificationResult<bool, float>>, ClassificationDecoding<bool>>()
            .WithOutputAllocation((input, out output) =>
            {
                output = new ClassificationResult<bool, float>[input.Length];
                return true;
            })
            .Build();

        await using ServiceProvider provider = services.BuildServiceProvider();
        var pipeline = provider.GetRequiredService<
            IPipeline<ReadOnlyMemory<string>, Memory<ClassificationResult<bool, float>>>>();
        ReadOnlyMemory<string> input = Enumerable.Range(0, 10)
            .Select(index => $"test {index}")
            .ToArray();
        Memory<ClassificationResult<bool, float>> output =
            await pipeline.ExecuteAsync(input, TestContext.Current.CancellationToken);

        output.ToArray().Should().HaveCount(10);
        output.ToArray().Should().OnlyContain(result => result.Choice);
    }
}
