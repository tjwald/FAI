using System.Text;
using FAI.Core.Extensions.DI;
using FAI.Core.Pipelines;
using FAI.NLP.Configuration;
using FAI.NLP.Tokenization;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.ML.Tokenizers;

namespace FAI.NLP.Extensions.DI.Tests;

public class FinitePipelineExtensionsTests
{
    [Fact]
    public async Task UseTokenCountOrdering_RestoresCallerOutputOrder()
    {
        ServiceProvider provider = BuildProvider(useOrdering: true, usePartitioning: false);
        var pipeline = provider.GetRequiredService<IPipeline<ReadOnlyMemory<TestTokenizable>, Memory<int>>>();
        var inner = provider.GetRequiredService<RecordingPipeline>();
        ReadOnlyMemory<TestTokenizable> input = new TestTokenizable[] { new(10), new(2), new(5) };
        Memory<int> output = await pipeline.ExecuteAsync(input, TestContext.Current.CancellationToken);

        Assert.Equal([2, 5, 10], inner.ObservedTokenCounts);
        Assert.Equal([10, 2, 5], output.ToArray());
    }

    [Fact]
    public async Task UseMaxPaddedTokensPartitioning_ExecutesExpectedRanges()
    {
        ServiceProvider provider = BuildProvider(useOrdering: false, usePartitioning: true);
        var pipeline = provider.GetRequiredService<IPipeline<ReadOnlyMemory<TestTokenizable>, Memory<int>>>();
        var inner = provider.GetRequiredService<RecordingPipeline>();
        ReadOnlyMemory<TestTokenizable> input = new TestTokenizable[] { new(4), new(4), new(4) };
        Memory<int> output = await pipeline.ExecuteAsync(input, TestContext.Current.CancellationToken);

        Assert.Equal([2, 1], inner.BatchSizes);
        Assert.Equal([4, 4, 4], output.ToArray());
    }

    [Fact]
    public async Task OrderAndPartition_ExecutesSortedPartitionsAndRestoresOutputOrder()
    {
        ServiceProvider provider = BuildProvider(useOrdering: true, usePartitioning: true);
        var pipeline = provider.GetRequiredService<IPipeline<ReadOnlyMemory<TestTokenizable>, Memory<int>>>();
        var inner = provider.GetRequiredService<RecordingPipeline>();
        ReadOnlyMemory<TestTokenizable> input = new TestTokenizable[] { new(9), new(2), new(4), new(3) };

        Memory<int> output = await pipeline.ExecuteAsync(input, TestContext.Current.CancellationToken);

        Assert.Equal([2, 3, 4, 9], inner.ObservedTokenCounts);
        Assert.Equal([2, 1, 1], inner.BatchSizes);
        Assert.Equal([9, 2, 4, 3], output.ToArray());
    }

    private static ServiceProvider BuildProvider(bool useOrdering, bool usePartitioning)
    {
        var services = new ServiceCollection();
        services.AddSingleton(CreateDummyTokenizer());
        services.AddSingleton(new TokenCountOrderingOptions(Ascending: true));
        services.AddSingleton(new MaxPaddedTokensPartitionerOptions(MaxPaddedTokenRatio: 0.5, MaxTokenCount: 10));
        PipelineBuilder<ReadOnlyMemory<TestTokenizable>, ReadOnlyMemory<TestTokenizable>> pipeline = services
            .AddPipeline<ReadOnlyMemory<TestTokenizable>>()
            .Then<ReadOnlyMemory<TestTokenizable>, PassThroughPipeline>();
        if (useOrdering && usePartitioning)
        {
            pipeline.UseTokenCountOrdering()
                .UseMaxPaddedTokensPartitioning()
                .Then<Memory<int>, RecordingPipeline>()
                .Build();
        }
        else if (useOrdering)
        {
            pipeline.UseTokenCountOrdering()
                .Then<Memory<int>, RecordingPipeline>()
                .Build();
        }
        else
        {
            pipeline.UseMaxPaddedTokensPartitioning()
                .Then<Memory<int>, RecordingPipeline>()
                .Build();
        }

        return services.BuildServiceProvider();
    }

    public sealed class PassThroughPipeline : IPipeline<ReadOnlyMemory<TestTokenizable>, ReadOnlyMemory<TestTokenizable>>
    {
        public ValueTask<ReadOnlyMemory<TestTokenizable>> ExecuteAsync(
            ReadOnlyMemory<TestTokenizable> input,
            CancellationToken cancellationToken = default)
            => ValueTask.FromResult(input);
    }

    public sealed class TestTokenizable(int tokenCount) : ITokenizable
    {
        public int TokenCount { get; } = tokenCount;
        public int MaxTokenLength => TokenCount;
        public int SentenceCount => 1;
    }

    public sealed class RecordingPipeline : IDestinationPipeline<ReadOnlyMemory<TestTokenizable>, Memory<int>>
    {
        public List<int> ObservedTokenCounts { get; } = [];
        public List<int> BatchSizes { get; } = [];

        public async ValueTask<Memory<int>> ExecuteAsync(
            ReadOnlyMemory<TestTokenizable> input,
            CancellationToken cancellationToken = default)
        {
            Memory<int> output = new int[input.Length];
            await ExecuteAsync(input, output, cancellationToken);
            return output;
        }

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
