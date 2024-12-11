using System.Runtime.InteropServices;
using Microsoft.ML.Tokenizers;
using ML.Infra.Abstractions;

namespace ML.Infra.PipelineBatchExecutors;

public readonly struct OutOfOrderBatchExecutor<TOutput> : IPipelineBatchExecutor<string, TOutput>
{
    private readonly Tokenizer _tokenizer;
    private readonly IPipelineBatchExecutor<string, TOutput> _executor;

    public OutOfOrderBatchExecutor(Tokenizer tokenizer, IPipelineBatchExecutor<string, TOutput> executor)
    {
        _tokenizer = tokenizer;
        _executor = executor;
    }

    public async Task ExecuteBatchPredict(IPipeline<string, TOutput> pipeline, ReadOnlyMemory<string> inputs, Memory<TOutput> outputSpan)
    {
        ReadOnlySpan<string> inputSpan = inputs.Span;
        int[] inputsSortedIndices = Enumerable.Range(0, inputSpan.Length).ToArray();
        string[] inputsSorted = inputs.Span.ToArray();
        
        var tokenComparer = new TokenCountComparer(_tokenizer);
        
        MemoryExtensions.Sort<string, int>(inputsSorted, inputsSortedIndices, tokenComparer.Compare);
        await _executor.ExecuteBatchPredict(pipeline, inputsSorted, outputSpan);
        MemoryExtensions.Sort<int, TOutput>(inputsSortedIndices, outputSpan.Span, (i1, i2) => i1.CompareTo(i2));
    }
}

file readonly struct TokenCountComparer
{
    private readonly Tokenizer _tokenizer;
    private readonly Dictionary<string, int> _counts;

    public TokenCountComparer(Tokenizer tokenizer)
    {
        _tokenizer = tokenizer;
        _counts = new Dictionary<string, int>();
    }

    public int Compare(string x, string y)
    {
        ref int xCount = ref CollectionsMarshal.GetValueRefOrAddDefault(_counts, x, out bool exists);
        if (!exists)
        {
            xCount = _tokenizer.CountTokens(x);
        }
        
        ref int yCount = ref CollectionsMarshal.GetValueRefOrAddDefault(_counts, x, out exists);
        if (!exists)
        {
            yCount = _tokenizer.CountTokens(y);
        }
        
        return xCount.CompareTo(yCount);
    }
}