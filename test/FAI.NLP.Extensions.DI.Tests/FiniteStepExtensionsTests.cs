using System.Text;
using FAI.Core.Extensions.DI;
using FAI.Core.Steps;
using FAI.NLP.Configuration;
using FAI.NLP.Tokenization;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.ML.Tokenizers;

namespace FAI.NLP.Extensions.DI.Tests;

public class FiniteStepExtensionsTests
{
    [Fact]
    public async Task UseTokenizingStep_TokenizesBeforeExecutingInnerStep()
    {
        ServiceProvider provider = BuildProvider(stage => stage.UseTokenizingStep());
        var pipeline = provider.GetRequiredService<IStep<ReadOnlyMemory<TestTokenizable>, Memory<int>>>();
        ReadOnlyMemory<TestTokenizable> input = new TestTokenizable[] { new(4), new(2) };
        var output = new int[input.Length];

        await pipeline.ExecuteAsync(input, output, TestContext.Current.CancellationToken);

        Assert.All(input.ToArray(), item => Assert.True(item.WasTokenized));
    }

    [Fact]
    public async Task UseTokenCountOrderingStep_RestoresCallerOutputOrder()
    {
        ServiceProvider provider = BuildProvider(stage => stage.UseTokenCountOrderingStep());
        var pipeline = provider.GetRequiredService<IStep<ReadOnlyMemory<TestTokenizable>, Memory<int>>>();
        var inner = provider.GetRequiredService<RecordingStep>();
        ReadOnlyMemory<TestTokenizable> input = new TestTokenizable[] { new(10), new(2), new(5) };
        var output = new int[input.Length];

        await pipeline.ExecuteAsync(input, output, TestContext.Current.CancellationToken);

        Assert.Equal([2, 5, 10], inner.ObservedTokenCounts);
        Assert.Equal([10, 2, 5], output);
    }

    [Fact]
    public async Task UseMaxPaddedTokensPartitioningStep_ExecutesExpectedRanges()
    {
        ServiceProvider provider = BuildProvider(stage => stage.UseMaxPaddedTokensPartitioningStep());
        var pipeline = provider.GetRequiredService<IStep<ReadOnlyMemory<TestTokenizable>, Memory<int>>>();
        var inner = provider.GetRequiredService<RecordingStep>();
        ReadOnlyMemory<TestTokenizable> input = new TestTokenizable[] { new(4), new(4), new(4) };
        var output = new int[input.Length];

        await pipeline.ExecuteAsync(input, output, TestContext.Current.CancellationToken);

        Assert.Equal([2, 1], inner.BatchSizes);
        Assert.Equal([4, 4, 4], output);
    }

    private static ServiceProvider BuildProvider(
        Action<PipelineStageBuilder<ReadOnlyMemory<TestTokenizable>, Memory<int>>> configure)
    {
        var services = new ServiceCollection();
        services.AddSingleton(CreateDummyTokenizer());
        services.AddSingleton(new TokenCountOrderingOptions(Ascending: true));
        services.AddSingleton(new MaxPaddedTokensPartitionerOptions(MaxPaddedTokenRatio: 0.5, MaxTokenCount: 10));
        services
            .AddPipeline<ReadOnlyMemory<TestTokenizable>>()
            .Then<Memory<int>, RecordingStep>(configure)
            .Build();
        return services.BuildServiceProvider();
    }

    public sealed class TestTokenizable(int tokenCount) : ITokenizable
    {
        public int TokenCount { get; } = tokenCount;
        public int MaxTokenLength => TokenCount;
        public int SentenceCount => 1;
        public bool WasTokenized { get; private set; }

        public void Tokenize(PretrainedTokenizer tokenizer)
        {
            WasTokenized = true;
        }
    }

    public sealed class RecordingStep : IAllocatingStep<ReadOnlyMemory<TestTokenizable>, Memory<int>>
    {
        public List<int> ObservedTokenCounts { get; } = [];
        public List<int> BatchSizes { get; } = [];

        public ValueTask<BatchLease<Memory<int>>> RentOutputAsync(
            ReadOnlyMemory<TestTokenizable> input,
            CancellationToken cancellationToken = default)
            => ValueTask.FromResult(new BatchLease<Memory<int>>(new int[input.Length]));

        public ValueTask ExecuteAsync(
            ReadOnlyMemory<TestTokenizable> input,
            Memory<int> output,
            CancellationToken cancellationToken = default)
        {
            BatchSizes.Add(input.Length);
            foreach (TestTokenizable item in input.Span)
            {
                ObservedTokenCounts.Add(item.TokenCount);
            }

            for (int index = 0; index < input.Length; index++)
            {
                output.Span[index] = input.Span[index].TokenCount;
            }

            return ValueTask.CompletedTask;
        }
    }

    private static PretrainedTokenizer CreateDummyTokenizer()
    {
        const string vocabulary = "[PAD]\n[CLS]\n[SEP]\n[MASK]\n[UNK]\n";
        using var stream = new MemoryStream(Encoding.UTF8.GetBytes(vocabulary));
        BertTokenizer tokenizer = BertTokenizer.Create(stream);
        var options = new PretrainedTokenizerOptions
        {
            MaxTokenLength = 128,
            PaddingToken = 0,
            TruncationOption = TruncationOption.Longest,
        };
        return new PretrainedTokenizer(tokenizer, options);
    }
}
