using ML.Infra.Abstractions;
using ML.NLP.Tokenization;

namespace ML.NLP.PipelineBatchExecutors;

public class MaxPaddedTokensBatchExecutor<TOutput> : IPipelineBatchExecutor<TokenizedText, TOutput>
{
    private readonly double _maxPaddedTokenRatio;
    private readonly int _maxTokenCount;
    private readonly IPipelineBatchExecutor<TokenizedText, TOutput> _executor;

    public MaxPaddedTokensBatchExecutor(IPipelineBatchExecutor<TokenizedText, TOutput> executor, double maxPaddedTokenRatio, int maxTokenCount)
    {
        _maxPaddedTokenRatio = maxPaddedTokenRatio;
        _maxTokenCount = maxTokenCount;
        _executor = executor;
    }

    public async Task ExecuteBatchPredict(ReadOnlyMemory<TokenizedText> inputs, Memory<TOutput> outputSpan)
    {
        IEnumerable<Range> ranges = GenerateRanges(inputs, _maxTokenCount, _maxPaddedTokenRatio);

        await Task.WhenAll(ranges.Select(range => _executor.ExecuteBatchPredict(inputs[range], outputSpan[range])));
    }

    private static IEnumerable<Range> GenerateRanges(ReadOnlyMemory<TokenizedText> inputs, int maxTokenCount, double maxPaddedTokenRatio)
    {
        int currentIndex = 0;
        // Use a float for micro-optimization in the ratio check
        float factor = 1.0f - (float)maxPaddedTokenRatio;

        while (currentIndex < inputs.Length)
        {
            ReadOnlySpan<TokenizedText> span = inputs.Span;

            int start = currentIndex;

            // Initialize the batch with the first candidate.
            int candidate = span[currentIndex].Count;
            int batchCount = 1;
            int batchSum = candidate;
            currentIndex++;

            // Process additional candidates until one violates a constraint.
            while (currentIndex < inputs.Length)
            {
                candidate = span[currentIndex].Count;
                int newBatchCount = batchCount + 1;
                int newPadded = newBatchCount * candidate;

                // Constraint 1: Check if the padded total exceeds the maximum allowed tokens.
                if (newPadded > maxTokenCount)
                    break;

                int newSum = batchSum + candidate;
                // Constraint 2: Check if the real token sum meets the ratio requirement.
                if (newSum < newPadded * factor)
                    break;

                // All constraints satisfied: accept the candidate.
                batchCount = newBatchCount;
                batchSum = newSum;
                currentIndex++;
            }

            yield return new Range(start, currentIndex);
        }
    }
}
