using FAI.Core.Steps;
using FAI.NLP.Configuration;
using FAI.NLP.Steps;
using FAI.NLP.Tests.Mocks;
using FAI.NLP.Tokenization;

namespace FAI.NLP.Tests.StepTests;

public class TokenBatchPolicyTests
{
    public class TestTokenizable(int tokenCount) : ITokenizable
    {
        public int TokenCount { get; set; } = tokenCount;
        public int MaxTokenLength => TokenCount;
        public int SentenceCount => 1;
        public void Tokenize(PretrainedTokenizer tokenizer) { }
    }

    [Fact]
    public void TokenCountOrdering_CreatesAscendingPermutation()
    {
        var ordering = new TokenCountOrdering<TestTokenizable>(new TokenCountOrderingOptions(Ascending: true));
        ReadOnlyMemory<TestTokenizable> inputs = new TestTokenizable[] { new(10), new(2), new(5) };

        int[] order = ordering.CreateOrder(inputs);

        Assert.Equal([1, 2, 0], order);
    }

    [Fact]
    public async Task TokenizingStep_TokenizesBeforeExecutingInnerStep()
    {
        var tokenizer = DummyTokenizerFactory.Create();
        var inner = new TokenCountStep();
        var step = new TokenizingStep<TokenizedText, int>(inner, tokenizer);
        ReadOnlyMemory<TokenizedText> inputs = new TokenizedText[] { new("hello"), new("hello world") };
        var output = new int[2];

        await step.ExecuteAsync(inputs, output, TestContext.Current.CancellationToken);

        Assert.All(inputs.ToArray(), input => Assert.NotNull(input.Tokens));
        Assert.Equal(inputs.ToArray().Select(input => input.TokenCount), output);
    }

    private sealed class TokenCountStep : IAllocatingStep<ReadOnlyMemory<TokenizedText>, Memory<int>>
    {
        public ValueTask<BatchLease<Memory<int>>> RentOutputAsync(
            ReadOnlyMemory<TokenizedText> input,
            CancellationToken cancellationToken = default)
            => ValueTask.FromResult(new BatchLease<Memory<int>>(new int[input.Length]));

        public ValueTask ExecuteAsync(
            ReadOnlyMemory<TokenizedText> input,
            Memory<int> output,
            CancellationToken cancellationToken = default)
        {
            for (int index = 0; index < input.Length; index++)
            {
                output.Span[index] = input.Span[index].TokenCount;
            }

            return ValueTask.CompletedTask;
        }
    }
}
