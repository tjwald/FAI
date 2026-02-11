using FAI.Core.Abstractions;
using FAI.NLP.Configuration.PipelineBatchExecutors;
using FAI.NLP.PipelineBatchExecutors;
using FAI.NLP.Tokenization;

namespace FAI.NLP.Tests.PipelineBatchExecutorTests;

public class TokenBatchSizeBatchExecutorTests
{
    public record TestTokenizable(int TokenCount) : ITokenizable
    {
        public int MaxTokenLength => TokenCount;
        public int SentenceCount => 1;
        public void Tokenize(PretrainedTokenizer tokenizer) { }
    }

    [Fact]
    public async Task ExecuteBatchPredict_SplitsIntoCorrectBatches()
    {
        // Arrange
        var mockExecutor = Substitute.For<IPipelineBatchExecutor<TestTokenizable, int>>();
        var options = new TokenBatchSizeBatchExecutorOptions { MaxTokensCount = 10 };
        var executor = new TokenBatchSizeBatchExecutor<TestTokenizable, int>(mockExecutor, options);

        // Inputs: 6, 5, 4, 7
        // Batch 1: [6]
        // Batch 2: [5, 4]
        // Batch 3: [7]
        TestTokenizable[] inputs = [new(6), new(5), new(4), new(7)];
        var outputs = new int[4];

        // Act
        await executor.ExecuteBatchPredict(inputs, outputs);

        // Assert
        // Distinguish Batch 1 and Batch 3 by using Received(2) or checking contents if possible without ref struct issues
        // Since we cannot easily use .Span in Arg.Is due to expression tree limitations, we check the number of matching calls.

        // One call with Length 2
        await mockExecutor.Received(1).ExecuteBatchPredict(
            Arg.Is<ReadOnlyMemory<TestTokenizable>>(m => m.Length == 2),
            Arg.Any<Memory<int>>());

        // Two calls with Length 1
        await mockExecutor.Received(2).ExecuteBatchPredict(
            Arg.Is<ReadOnlyMemory<TestTokenizable>>(m => m.Length == 1),
            Arg.Any<Memory<int>>());
    }
}
