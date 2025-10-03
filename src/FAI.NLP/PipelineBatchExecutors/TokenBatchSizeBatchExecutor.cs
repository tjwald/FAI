using FAI.Core.Abstractions;
using FAI.NLP.Configuration.PipelineBatchExecutors;
using FAI.NLP.Tokenization;

namespace FAI.NLP.PipelineBatchExecutors;

/// <summary>
/// Represents a batch executor that processes inputs based on the total token count allowed in each batch.
/// </summary>
/// <typeparam name="TToken">The type of the tokenizable input items.</typeparam>
/// <typeparam name="TOutput">The type of the output items.</typeparam>
public class TokenBatchSizeBatchExecutor<TToken, TOutput> : IPipelineBatchExecutor<TToken, TOutput> where TToken : ITokenizable
{
    private readonly int _maxTokenCount;
    private readonly IPipelineBatchExecutor<TToken, TOutput> _executor;

    public TokenBatchSizeBatchExecutor(IPipelineBatchExecutor<TToken, TOutput> executor, TokenBatchSizeBatchExecutorOptions options)
        : this(executor, options.MaxTokensCount) { }


    /// <summary>
    /// Initializes a new instance of the <see cref="TokenBatchSizeBatchExecutor{TToken, TOutput}"/> class.
    /// </summary>
    /// <param name="executor">The underlying pipeline batch executor used to handle prediction tasks.</param>
    /// <param name="maxTokenCount">The maximum number of tokens allowed per batch.</param>
    public TokenBatchSizeBatchExecutor(IPipelineBatchExecutor<TToken, TOutput> executor, int maxTokenCount)
    {
        _maxTokenCount = maxTokenCount;
        _executor = executor;
    }

    /// <summary>
    /// Executes batch prediction asynchronously by splitting the input into ranges
    /// that meet the token count constraint and invoking the underlying executor on each range.
    /// </summary>
    /// <param name="inputs">The input batch of tokenizable items.</param>
    /// <param name="outputSpan">The memory span for storing output results.</param>
    /// <returns>A task representing the asynchronous batch prediction operation.</returns>
    public async Task ExecuteBatchPredict(ReadOnlyMemory<TToken> inputs, Memory<TOutput> outputSpan)
    {
        IEnumerable<Range> ranges = GenerateRanges(inputs, _maxTokenCount);

        await Task.WhenAll(ranges.Select(range => _executor.ExecuteBatchPredict(inputs[range], outputSpan[range])));
    }

    /// <summary>
    /// Generates ranges of input items for batching, ensuring the token count constraint is met.
    /// </summary>
    /// <param name="inputs">The input batch of tokenizable items.</param>
    /// <param name="maxTokenCount">The maximum number of tokens allowed per batch.</param>
    /// <returns>An enumerable of ranges representing valid batches.</returns>
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
