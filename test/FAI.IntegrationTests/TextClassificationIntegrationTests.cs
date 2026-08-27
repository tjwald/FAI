namespace FAI.IntegrationTests;

public class TextClassificationIntegrationTests
{
    [Fact]
    public async Task FullPipeline_ShouldClassifyText()
    {
        var services = new ServiceCollection();
        services.AddSingleton(new ClassificationOptions<bool>([false, true]));
        services.AddSingleton(DummyTokenizerFactory.Create());
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
                stage => stage.UseTokenizingStep())
            .Build();

        await using ServiceProvider provider = services.BuildServiceProvider();
        var pipeline = provider.GetRequiredService<
            IStep<ReadOnlyMemory<TokenizedText>, Memory<ClassificationResult<bool, float>>>>();
        ReadOnlyMemory<TokenizedText> input = new TokenizedText[] { new("hello") };
        var output = new ClassificationResult<bool, float>[1];

        await pipeline.ExecuteAsync(input, output, TestContext.Current.CancellationToken);

        output.Should().HaveCount(1);
        output[0].Choice.Should().BeTrue();
    }
}
