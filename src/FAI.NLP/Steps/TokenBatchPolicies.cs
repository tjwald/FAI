using FAI.Core.Steps;
using FAI.NLP.Configuration;
using FAI.NLP.Tokenization;

namespace FAI.NLP.Steps;

public sealed class TokenizingStep<TInput, TOutput> :
    IStep<ReadOnlyMemory<TInput>, Memory<TOutput>>
    where TInput : ITokenizable
{
    private readonly IStep<ReadOnlyMemory<TInput>, Memory<TOutput>> _inner;
    private readonly PretrainedTokenizer _tokenizer;

    public TokenizingStep(
        IStep<ReadOnlyMemory<TInput>, Memory<TOutput>> inner,
        PretrainedTokenizer tokenizer)
    {
        _inner = inner;
        _tokenizer = tokenizer;
    }

    public async ValueTask<Memory<TOutput>> ExecuteAsync(
        ReadOnlyMemory<TInput> input,
        CancellationToken cancellationToken = default)
    {
        var parallelOptions = new ParallelOptions { CancellationToken = cancellationToken };
        Parallel.ForEach(input.ToArray(), parallelOptions, item => item.Tokenize(_tokenizer));
        return await _inner.ExecuteAsync(input, cancellationToken);
    }
}

public sealed class TokenCountOrdering<TInput> : IIndexOrdering<ReadOnlyMemory<TInput>>
    where TInput : ITokenizable
{
    private readonly bool _ascending;

    public TokenCountOrdering(TokenCountOrderingOptions options)
    {
        _ascending = options.Ascending;
    }

    public int[] CreateOrder(ReadOnlyMemory<TInput> batch)
    {
        int[] indices = Enumerable.Range(0, batch.Length).ToArray();
        Array.Sort(indices, (left, right) =>
        {
            int comparison = batch.Span[left].MaxTokenLength.CompareTo(batch.Span[right].MaxTokenLength);
            if (!_ascending)
            {
                comparison = -comparison;
            }

            return comparison != 0 ? comparison : left.CompareTo(right);
        });
        return indices;
    }
}

public sealed class MaxPaddedTokensPartitioner<TInput> : IBatchPartitioner<ReadOnlyMemory<TInput>>
    where TInput : ITokenizable
{
    private readonly MaxPaddedTokensPartitionerOptions _options;

    public MaxPaddedTokensPartitioner(MaxPaddedTokensPartitionerOptions options)
    {
        _options = options;
    }

    public IEnumerable<Range> Partition(ReadOnlyMemory<TInput> batch)
    {
        int currentIndex = 0;
        float factor = 1.0f - (float)_options.MaxPaddedTokenRatio;

        while (currentIndex < batch.Length)
        {
            int start = currentIndex;
            int candidate = batch.Span[currentIndex].TokenCount;
            int batchCount = 1;
            int batchSum = candidate;
            currentIndex++;

            while (currentIndex < batch.Length)
            {
                TInput current = batch.Span[currentIndex];
                candidate = current.MaxTokenLength;
                int newBatchCount = batchCount + current.SentenceCount;
                int newPadded = newBatchCount * candidate;
                if (newPadded > _options.MaxTokenCount)
                {
                    break;
                }

                int newSum = batchSum + candidate;
                if (newSum < newPadded * factor)
                {
                    break;
                }

                batchCount = newBatchCount;
                batchSum = newSum;
                currentIndex++;
            }

            yield return start..currentIndex;
        }
    }
}
