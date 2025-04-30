using ML.Infra.Abstractions;
using ML.NLP.Tokenization;

namespace ML.NLP.PipelineBatchExecutors;

/// <summary>
/// A batch executor that sorts tokenized inputs by token count before executing batch predictions.
/// </summary>
/// <typeparam name="TToken">The type of tokenizable input items.</typeparam>
/// <typeparam name="TOutput">The type of output items.</typeparam>
public class TokenCountSortingBatchExecutor<TToken, TOutput> : IPipelineBatchExecutor<TToken, TOutput> where TToken : ITokenizable
{
    private readonly PretrainedTokenizer _tokenizer;
    private readonly IPipelineBatchExecutor<TToken, TOutput> _executor;

    /// <summary>
    /// Initializes a new instance of the <see cref="TokenCountSortingBatchExecutor{TToken, TOutput}"/> class.
    /// </summary>
    /// <param name="executor">The underlying batch executor responsible for predictions.</param>
    /// <param name="tokenizer">The tokenizer used for tokenizing input items.</param>
    public TokenCountSortingBatchExecutor(IPipelineBatchExecutor<TToken, TOutput> executor, PretrainedTokenizer tokenizer)
    {
        _tokenizer = tokenizer;
        _executor = executor;
    }

    /// <summary>
    /// Executes batch prediction asynchronously by first sorting tokenized inputs by token count
    /// and then calling the underlying executor on the sorted inputs.
    /// </summary>
    /// <param name="inputs">The input batch of tokenizable items.</param>
    /// <param name="outputSpan">The memory span for storing output results.</param>
    /// <returns>A task representing the asynchronous batch prediction operation.</returns>
    public async Task ExecuteBatchPredict(ReadOnlyMemory<TToken> inputs, Memory<TOutput> outputSpan)
    {
        ReadOnlySpan<TToken> inputSpan = inputs.Span;
        int[] inputsSortedIndices = Enumerable.Range(0, inputSpan.Length).ToArray();
        TToken[] inputsSorted = inputs.Span.ToArray();

        foreach (var input in inputsSorted)
        {
            input.Tokenize(_tokenizer);
        }
        
        var tokenComparer = new TokenCountComparer<TToken>();

        MemoryExtensions.Sort<TToken, int, TokenCountComparer<TToken>>(inputsSorted, inputsSortedIndices, tokenComparer);
        await _executor.ExecuteBatchPredict(inputsSorted, outputSpan);
        MemoryExtensions.Sort<int, TOutput>(inputsSortedIndices, outputSpan.Span);
    }
}
