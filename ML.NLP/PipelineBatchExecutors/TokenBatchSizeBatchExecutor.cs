using ML.Infra.Abstractions;
using ML.NLP.Tokenization;

namespace ML.NLP.PipelineBatchExecutors;

public class TokenBatchSizeBatchExecutor<TToken, TOutput>: IPipelineBatchExecutor<TToken, TOutput> where TToken: ITokenizable
{
    private readonly int _maxTokenCount;
    private readonly IPipelineBatchExecutor<TToken, TOutput> _executor;

    public TokenBatchSizeBatchExecutor(IPipelineBatchExecutor<TToken, TOutput> executor, int maxTokenCount)
    {
        _maxTokenCount = maxTokenCount;
        _executor = executor;
    }
    
    public async Task ExecuteBatchPredict(ReadOnlyMemory<TToken> inputs, Memory<TOutput> outputSpan)
    {
        IEnumerable<Range> ranges = GenerateRanges(inputs, _maxTokenCount);
        
        await Task.WhenAll(ranges.Select(range => _executor.ExecuteBatchPredict(inputs[range], outputSpan[range])));
    }

    private static IEnumerable<Range> GenerateRanges(ReadOnlyMemory<TToken> inputs, int maxTokenCount)
    {
        int currentIndex = 0;

        while (currentIndex < inputs.Length)
        {
            int tokenCount = 0;
            int start = currentIndex;

            while (currentIndex < inputs.Length && tokenCount + inputs.Span[currentIndex].TokenCount <= maxTokenCount)
            {
                tokenCount += inputs.Span[currentIndex].TokenCount;
                currentIndex++;
            }

            var range = new Range(start, currentIndex);
            yield return range;
        }
    }
}