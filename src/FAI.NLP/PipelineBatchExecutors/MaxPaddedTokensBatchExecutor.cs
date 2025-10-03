using FAI.Core.Abstractions;
using FAI.NLP.Configuration.PipelineBatchExecutors;
using FAI.NLP.Tokenization;

namespace FAI.NLP.PipelineBatchExecutors;

/// <summary>
/// Represents a batch executor that limits the number of padded tokens in each batch
/// by enforcing constraints on token count and padding ratio.
/// </summary>
/// <typeparam name="TToken">The type of the tokenizable input items.</typeparam>
/// <typeparam name="TOutput">The type of the output items.</typeparam>
public class MaxPaddedTokensBatchExecutor<TToken, TOutput> : IPipelineBatchExecutor<TToken, TOutput> where TToken : ITokenizable
{
    private readonly double _maxPaddedTokenRatio;
    private readonly int _maxTokenCount;
    private readonly IPipelineBatchExecutor<TToken, TOutput> _executor;

    /// <summary>
    /// Initializes a new instance of the <see cref="MaxPaddedTokensBatchExecutor{TToken, TOutput}"/> class.
    /// </summary>
    /// <param name="executor">The underlying pipeline batch executor to handle actual prediction tasks.</param>
    /// <param name="maxPaddedTokenRatio">
    /// The maximum allowed ratio of padded tokens to actual tokens in a batch.
    /// </param>
    /// <param name="maxTokenCount">The maximum number of tokens allowed per batch.</param>
    public MaxPaddedTokensBatchExecutor(IPipelineBatchExecutor<TToken, TOutput> executor, double maxPaddedTokenRatio, int maxTokenCount)
    {
        _maxPaddedTokenRatio = maxPaddedTokenRatio;
        _maxTokenCount = maxTokenCount;
        _executor = executor;
    }

    public MaxPaddedTokensBatchExecutor(IPipelineBatchExecutor<TToken, TOutput> executor, MaxPaddedTokensBatchExecutorOptions options) : this(executor, options.MaxPaddedTokenRatio, options.MaxTokenCount)
    { }

    /// <summary>
    /// Executes batch prediction asynchronously by splitting the input into valid ranges
    /// based on token constraints and calling the underlying executor on each range.
    /// </summary>
    /// <param name="inputs">The input batch of tokenizable items.</param>
    /// <param name="outputSpan">The memory span for storing output results.</param>
    /// <returns>A task that represents the asynchronous batch prediction operation.</returns>
    public async Task ExecuteBatchPredict(ReadOnlyMemory<TToken> inputs, Memory<TOutput> outputSpan)
    {
        IEnumerable<Range> ranges = GenerateRanges(inputs, _maxTokenCount, _maxPaddedTokenRatio);

        await Task.WhenAll(ranges.Select(range => _executor.ExecuteBatchPredict(inputs[range], outputSpan[range])));
    }

    /// <summary>
    /// Generates ranges of input items for batching, ensuring token count and padding ratio constraints are met.
    /// </summary>
    /// <param name="inputs">The input batch of tokenizable items.</param>
    /// <param name="maxTokenCount">The maximum number of tokens allowed per batch.</param>
    /// <param name="maxPaddedTokenRatio">
    /// The maximum allowed ratio of padded tokens to actual tokens in a batch.
    /// </param>
    /// <returns>An enumerable of ranges representing valid batches.</returns>
    private static IEnumerable<Range> GenerateRanges(ReadOnlyMemory<TToken> inputs, int maxTokenCount, double maxPaddedTokenRatio)
    {
        int currentIndex = 0;
        // Use a float for micro-optimization in the ratio check
        float factor = 1.0f - (float)maxPaddedTokenRatio;

        while (currentIndex < inputs.Length)
        {
            ReadOnlySpan<TToken> span = inputs.Span;

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
