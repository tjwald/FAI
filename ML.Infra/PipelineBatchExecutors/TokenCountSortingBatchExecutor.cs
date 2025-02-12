using ML.Infra.Abstractions;
using ML.Infra.Tokenization;

namespace ML.Infra.PipelineBatchExecutors;

public readonly struct TokenCountSortingBatchExecutor<TOutput> : IPipelineBatchExecutor<TokenizedText, TOutput>
{
    private readonly PretrainedTokenizer _tokenizer;
    private readonly IPipelineBatchExecutor<TokenizedText, TOutput> _executor;

    public TokenCountSortingBatchExecutor(PretrainedTokenizer tokenizer, IPipelineBatchExecutor<TokenizedText, TOutput> executor)
    {
        _tokenizer = tokenizer;
        _executor = executor;
    }

    public async Task ExecuteBatchPredict(IPipeline<TokenizedText, TOutput> pipeline, ReadOnlyMemory<TokenizedText> inputs, Memory<TOutput> outputSpan)
    {
        ReadOnlySpan<TokenizedText> inputSpan = inputs.Span;
        int[] inputsSortedIndices = Enumerable.Range(0, inputSpan.Length).ToArray();
        TokenizedText[] inputsSorted = inputs.Span.ToArray();

        var tokenComparer = new TokenCountComparer(_tokenizer);

        MemoryExtensions.Sort<TokenizedText, int, TokenCountComparer>(inputsSorted, inputsSortedIndices, tokenComparer);
        await _executor.ExecuteBatchPredict(pipeline, inputsSorted, outputSpan);
        MemoryExtensions.Sort<int, TOutput>(inputsSortedIndices, outputSpan.Span);
    }
}

file readonly struct TokenCountComparer : IComparer<TokenizedText>
{
    private readonly PretrainedTokenizer _tokenizer;

    public TokenCountComparer(PretrainedTokenizer tokenizer)
    {
        _tokenizer = tokenizer;
    }

    public int Compare(TokenizedText? x, TokenizedText? y)
    {
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(y);

        x.Tokens ??= _tokenizer.Tokenize(x.Text);
        y.Tokens ??= _tokenizer.Tokenize(y.Text);

        int xCount = x.Count;
        int yCount = y.Count;

        return xCount.CompareTo(yCount);
    }
}