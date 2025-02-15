using ML.Infra.Abstractions;
using ML.NLP.Tokenization;

namespace ML.NLP.PipelineBatchExecutors;

public readonly struct TokenCountSortingBatchExecutor<TOutput> : IPipelineBatchExecutor<TokenizedText, TOutput>
{
    private readonly PretrainedTokenizer _tokenizer;
    private readonly IPipelineBatchExecutor<TokenizedText, TOutput> _executor;

    public TokenCountSortingBatchExecutor(PretrainedTokenizer tokenizer, IPipelineBatchExecutor<TokenizedText, TOutput> executor)
    {
        _tokenizer = tokenizer;
        _executor = executor;
    }

    public async Task ExecuteBatchPredict(ReadOnlyMemory<TokenizedText> inputs, Memory<TOutput> outputSpan)
    {
        ReadOnlySpan<TokenizedText> inputSpan = inputs.Span;
        int[] inputsSortedIndices = Enumerable.Range(0, inputSpan.Length).ToArray();
        TokenizedText[] inputsSorted = inputs.Span.ToArray();

        var tokenComparer = new TokenCountComparer(_tokenizer);

        MemoryExtensions.Sort<TokenizedText, int, TokenCountComparer>(inputsSorted, inputsSortedIndices, tokenComparer);
        await _executor.ExecuteBatchPredict(inputsSorted, outputSpan);
        MemoryExtensions.Sort<int, TOutput>(inputsSortedIndices, outputSpan.Span);
    }
}