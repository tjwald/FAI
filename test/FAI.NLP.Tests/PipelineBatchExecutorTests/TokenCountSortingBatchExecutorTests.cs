using FAI.Core.Abstractions;
using FAI.NLP.Configuration.PipelineBatchExecutors;
using FAI.NLP.PipelineBatchExecutors;
using FAI.NLP.Tests.Mocks;
using FAI.NLP.Tokenization;

namespace FAI.NLP.Tests.PipelineBatchExecutorTests;

public class TokenCountSortingBatchExecutorTests
{
    public class TestTokenizable(int tokenCount) : ITokenizable
    {
        public int TokenCount { get; set; } = tokenCount;
        public int MaxTokenLength => TokenCount;
        public int SentenceCount => 1;
        public void Tokenize(PretrainedTokenizer tokenizer) { }
    }

    [Fact]
    public async Task ExecuteBatchPredict_SortsAndUnsortsCorrectly()
    {
        // Arrange
        var mockExecutor = Substitute.For<IPipelineBatchExecutor<TestTokenizable, int>>();
        var tokenizer = DummyTokenizerFactory.Create();
        var options = new TokenCountSortingBatchExecutorOptions { Ascending = true };
        var executor = new TokenCountSortingBatchExecutor<TestTokenizable, int>(mockExecutor, tokenizer, options);

        var inputs = new TestTokenizable[] { new(10), new(2), new(5) };
        var outputs = new int[3];

        // We expect the executor to receive [new(2), new(5), new(10)]
        // We will mock the response of the inner executor to return [22, 55, 1010] (indices: 0->22, 1->55, 2->1010)
        // Since input was [10, 2, 5], final output should be [1010, 22, 55]
        mockExecutor.ExecuteBatchPredict(Arg.Any<ReadOnlyMemory<TestTokenizable>>(), Arg.Any<Memory<int>>())
            .Returns(x =>
            {
                var inputMem = x.ArgAt<ReadOnlyMemory<TestTokenizable>>(0);
                var outputMem = x.ArgAt<Memory<int>>(1);
                for (int i = 0; i < inputMem.Length; i++)
                {
                    outputMem.Span[i] = inputMem.Span[i].TokenCount * 11; // Dummy operation
                }
                return Task.CompletedTask;
            });

        // Act
        await executor.ExecuteBatchPredict(inputs, outputs);

        // Assert
        Assert.Equal(110, outputs[0]); // 10 * 11
        Assert.Equal(22, outputs[1]);  // 2 * 11
        Assert.Equal(55, outputs[2]);  // 5 * 11
    }
}
