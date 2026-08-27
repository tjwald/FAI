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
        services.AddSingleton<IBorrowedTensorProducer<Tensor<long>[], float>>(
            new LogicalMockModelStep([[0.1f, 0.9f]]));
        services.AddSingleton<ClassificationDecodingStep<bool>>();
        services
            .AddPipeline<ReadOnlyMemory<TokenizedText>>()
            .Then(
                pipeline => pipeline
                    .Then<Tensor<long>[], TextBatchEncodingStep>()
                    .ThenBorrowed(
                        sp => sp.GetRequiredService<IBorrowedTensorProducer<Tensor<long>[], float>>(),
                        sp => sp.GetRequiredService<ClassificationDecodingStep<bool>>(),
                        (_, input, _) => ValueTask.FromResult(
                            new BatchLease<Memory<ClassificationResult<bool, float>>>(
                                new ClassificationResult<bool, float>[input[0].Lengths[0]]))),
                (_, input, _) => ValueTask.FromResult(
                    new BatchLease<Memory<ClassificationResult<bool, float>>>(
                        new ClassificationResult<bool, float>[input.Length])),
                stage => stage
                    .UseTokenizingStep()
                    .UseTokenCountOrderingStep()
                    .UseMaxPaddedTokensPartitioningStep())
            .Build();

        await using ServiceProvider provider = services.BuildServiceProvider();
        var pipeline = provider.GetRequiredService<
            IStep<ReadOnlyMemory<TokenizedText>, Memory<ClassificationResult<bool, float>>>>();
        ReadOnlyMemory<TokenizedText> input = Enumerable.Range(0, 10)
            .Select(index => new TokenizedText($"test {index}"))
            .ToArray();
        var output = new ClassificationResult<bool, float>[input.Length];

        await pipeline.ExecuteAsync(input, output, TestContext.Current.CancellationToken);

        output.Should().HaveCount(10);
        output.Should().OnlyContain(result => result.Choice);
    }
}
