using FAI.Core.Abstractions;
using FAI.NLP.PipelineBatchExecutors;
using FAI.NLP.Tokenization;

namespace FAI.NLP.Configuration.PipelineBatchExecutors;

public class TokenBasedBatchExecutorBuilder<TToken, TOutput>
    : TokenPipelineBatchExecutorBuilder<TToken, TOutput, TokenBasedBatchExecutorBuilder<TToken, TOutput>>
    where TToken : ITokenizable
{
    public bool SortTokens { get; set; } = true;
    public int? MaxTokensCount { get; set; } = null;

    public override async ValueTask<IPipelineBatchExecutor<TToken, TOutput>> BuildAsync()
    {
        IPipelineBatchExecutor<TToken, TOutput> executor = await CreateInternalPipelineBatchExecutorAsync();
        if (MaxTokensCount.HasValue)
        {
            Console.WriteLine("Using TokenBatchSize chunking");
            executor = new TokenBatchSizeBatchExecutor<TToken, TOutput>(executor, MaxTokensCount.Value);
        }

        if (SortTokens)
        {
            Console.WriteLine("Using Sort by token count execution");
            executor = new TokenCountSortingBatchExecutor<TToken, TOutput>(executor, await GetTokenizer());
        }

        return executor;
    }
}