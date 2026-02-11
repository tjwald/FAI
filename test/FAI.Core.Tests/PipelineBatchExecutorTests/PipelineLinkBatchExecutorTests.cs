using System.Buffers;
using FAI.Core.Abstractions;
using FAI.Core.PipelineBatchExecutors;
using NSubstitute;

namespace FAI.Core.Tests.PipelineBatchExecutorTests;

public class PipelineLinkBatchExecutorTests
{
    [Fact]
    public async Task ExecuteBatchPredict_AppliesMappingFunctionAndDelegatesToNext()
    {
        // Arrange
        var nextPipeline = Substitute.For<IPipeline<string, int>>();
        Func<int, string> mapping = i => i.ToString();
        var pool = ArrayPool<string>.Shared;
        var executor = new PipelineLinkBatchExecutor<int, string, int>(nextPipeline, mapping, pool);

        var inputs = new[] { 1, 2, 3 }.AsMemory();
        var outputs = new int[3].AsMemory();

        // Act
        await executor.ExecuteBatchPredict(inputs, outputs);

        // Assert
        var expectedArray = (string[])["1", "2", "3"];
        await nextPipeline.Received(1).BatchPredict(
            Arg.Is<ReadOnlyMemory<string>>(m => m.ToArray().SequenceEqual(expectedArray)),
            Arg.Is<Memory<int>>(m => m.Length == 3));
    }
}
