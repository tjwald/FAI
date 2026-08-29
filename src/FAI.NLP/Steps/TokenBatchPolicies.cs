using FAI.Core.Steps;
using FAI.NLP.Configuration;
using FAI.NLP.InferenceTasks.TextMultipleChoice;
using FAI.NLP.Tokenization;

namespace FAI.NLP.Steps;

public sealed class TextTokenizationStep :
    IStep<ReadOnlyMemory<string>, ReadOnlyMemory<TokenizedText>>
{
    private readonly PretrainedTokenizer _tokenizer;

    public TextTokenizationStep(PretrainedTokenizer tokenizer)
    {
        _tokenizer = tokenizer;
    }

    public ValueTask<ReadOnlyMemory<TokenizedText>> ExecuteAsync(
        ReadOnlyMemory<string> input,
        CancellationToken cancellationToken = default)
    {
        var output = new TokenizedText[input.Length];
        Parallel.For(0, input.Length, new ParallelOptions { CancellationToken = cancellationToken }, index =>
        {
            string text = input.Span[index];
            output[index] = new TokenizedText(text, _tokenizer.Tokenize(text).ToArray());
        });
        return ValueTask.FromResult<ReadOnlyMemory<TokenizedText>>(output);
    }
}

public sealed class TextMultipleChoiceTokenizationStep :
    IStep<ReadOnlyMemory<TextMultipleChoiceInput>, ReadOnlyMemory<TokenizedTextMultipleChoiceInput>>
{
    private readonly PretrainedTokenizer _tokenizer;

    public TextMultipleChoiceTokenizationStep(PretrainedTokenizer tokenizer)
    {
        _tokenizer = tokenizer;
    }

    public ValueTask<ReadOnlyMemory<TokenizedTextMultipleChoiceInput>> ExecuteAsync(
        ReadOnlyMemory<TextMultipleChoiceInput> input,
        CancellationToken cancellationToken = default)
    {
        var output = new TokenizedTextMultipleChoiceInput[input.Length];
        Parallel.For(0, input.Length, new ParallelOptions { CancellationToken = cancellationToken }, inputIndex =>
        {
            TextMultipleChoiceInput item = input.Span[inputIndex];
            var choices = new TokenizedText[item.Choices.Length];
            for (int choiceIndex = 0; choiceIndex < choices.Length; choiceIndex++)
            {
                string choice = item.Choices[choiceIndex];
                choices[choiceIndex] = new TokenizedText(
                    choice,
                    _tokenizer.Tokenize(item.Context, choice).ToArray());
            }

            output[inputIndex] = new TokenizedTextMultipleChoiceInput(item.Context, choices);
        });
        return ValueTask.FromResult<ReadOnlyMemory<TokenizedTextMultipleChoiceInput>>(output);
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
