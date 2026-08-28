namespace FAI.IntegrationTests;

public class TextClassificationIntegrationTests
{
    [Fact]
    public async Task FullPipeline_ShouldClassifyText()
    {
        var services = new ServiceCollection();
        services.AddSingleton(new ClassificationOptions<bool>([false, true]));
        services.AddSingleton(DummyTokenizerFactory.Create());
        services.AddSingleton<IStep<Tensor<long>[], TensorOutputs<float>>>(
            new LogicalMockModelStep([[0.1f, 0.9f]]));
        services.AddSingleton<ClassificationDecodingStep<bool>>();
        services
            .AddPipeline<ReadOnlyMemory<TokenizedText>>()
            .Then(
                pipeline => pipeline
                    .Then<Tensor<long>[], TextBatchEncodingStep>()
                    .Then(sp =>
                        sp.GetRequiredService<IStep<Tensor<long>[], TensorOutputs<float>>>())
                    .Then<Memory<ClassificationResult<bool, float>>, ClassificationDecodingStep<bool>>(),
                stage => stage.UseTokenizingStep())
            .Build();

        await using ServiceProvider provider = services.BuildServiceProvider();
        var pipeline = provider.GetRequiredService<
            IStep<ReadOnlyMemory<TokenizedText>, Memory<ClassificationResult<bool, float>>>>();
        ReadOnlyMemory<TokenizedText> input = new TokenizedText[] { new("hello") };
        Memory<ClassificationResult<bool, float>> output =
            await pipeline.ExecuteAsync(input, TestContext.Current.CancellationToken);

        output.ToArray().Should().HaveCount(1);
        output.Span[0].Choice.Should().BeTrue();
    }
}
