using ML.Infra.Abstractions;
using ML.NLP.Tokenization;

namespace ML.NLP.PipelineBatchExecutors;

public class TokenBatchSizeBatchExecutor<TOutput>: IPipelineBatchExecutor<TokenizedText, TOutput>
{
    private readonly int _maxTokenCount;
    private readonly IPipelineBatchExecutor<TokenizedText, TOutput> _executor;

    public TokenBatchSizeBatchExecutor(IPipelineBatchExecutor<TokenizedText, TOutput> executor, int maxTokenCount)
    {
        _maxTokenCount = maxTokenCount;
        _executor = executor;
    }
    
    public async Task ExecuteBatchPredict(ReadOnlyMemory<TokenizedText> inputs, Memory<TOutput> outputSpan)
    {
        IEnumerable<Range> ranges = GenerateRanges(inputs, _maxTokenCount);
        
        await Task.WhenAll(ranges.Select(range => _executor.ExecuteBatchPredict(inputs[range], outputSpan[range])));
    }

    private static IEnumerable<Range> GenerateRanges(ReadOnlyMemory<TokenizedText> inputs, int maxTokenCount)
    {
        int currentIndex = 0;

        while (currentIndex < inputs.Length)
        {
            int tokenCount = 0;
            int start = currentIndex;

            while (currentIndex < inputs.Length && tokenCount + inputs.Span[currentIndex].Count <= maxTokenCount)
            {
                tokenCount += inputs.Span[currentIndex].Count;
                currentIndex++;
            }

            var range = new Range(start, currentIndex);
            yield return range;
        }
    }

}