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
    public async Task TextTokenizationStep_ReturnsImmutableTokenizedBatch()
    {
        var tokenizer = DummyTokenizerFactory.Create();
        var step = new TextTokenizationStep(tokenizer);
        ReadOnlyMemory<string> inputs = new string[] { "hello", "hello world" };

        ReadOnlyMemory<TokenizedText> output = await step.ExecuteAsync(inputs, TestContext.Current.CancellationToken);

        Assert.Equal(inputs.ToArray(), output.ToArray().Select(item => item.Text));
        Assert.All(output.ToArray(), item => Assert.True(item.TokenCount > 0));
    }
}
