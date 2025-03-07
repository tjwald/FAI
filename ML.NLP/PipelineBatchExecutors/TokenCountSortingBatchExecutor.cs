using ML.Infra.Abstractions;
using ML.NLP.Tokenization;

namespace ML.NLP.PipelineBatchExecutors;

public class TokenCountSortingBatchExecutor<TToken, TOutput> : IPipelineBatchExecutor<TToken, TOutput> where TToken: ITokenizable
{
    private readonly PretrainedTokenizer _tokenizer;
    private readonly IPipelineBatchExecutor<TToken, TOutput> _executor;

    public TokenCountSortingBatchExecutor(PretrainedTokenizer tokenizer, IPipelineBatchExecutor<TToken, TOutput> executor)
    {
        _tokenizer = tokenizer;
        _executor = executor;
    }

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