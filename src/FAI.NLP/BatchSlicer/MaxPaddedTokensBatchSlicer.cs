using FAI.Core.Abstractions;
using FAI.NLP.Configuration.PipelineBatchExecutors;
using FAI.NLP.Tokenization;

namespace FAI.NLP.BatchSlicer;

public sealed class MaxPaddedTokensBatchSlicer<TInput> : IBatchSlicer<TInput> where TInput : ITokenizable
{
    private readonly MaxPaddedTokensSlicerOptions _options;

    public MaxPaddedTokensBatchSlicer(MaxPaddedTokensSlicerOptions options)
    {
        this._options = options;
    }

    public IEnumerable<Range> Slice(ReadOnlyMemory<TInput> inputs)
    {
        int currentIndex = 0;
        // Use a float for micro-optimization in the ratio check
        float factor = 1.0f - (float)_options.MaxPaddedTokenRatio;

        while (currentIndex < inputs.Length)
        {
            ReadOnlySpan<TInput> span = inputs.Span;

            int start = currentIndex;

            // Initialize the batch with the first candidate.
            int candidate = span[currentIndex].TokenCount;
            int batchCount = 1;
            int batchSum = candidate;
            currentIndex++;

            // Process additional candidates until one violates a constraint.
            while (currentIndex < inputs.Length)
            {
                var current = span[currentIndex];
                candidate = current.MaxTokenLength;
                int newBatchCount = batchCount + current.SentenceCount;
                int newPadded = newBatchCount * candidate; // assume input is sorted by MaxTokenLength

                // Constraint 1: Check if the padded total exceeds the maximum allowed tokens.
                if (newPadded > _options.MaxTokenCount)
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
