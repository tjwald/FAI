using FAI.Core.Abstractions;
using FAI.NLP.PipelineBatchExecutors;
using FAI.NLP.Tokenization;

namespace FAI.NLP.Configuration.PipelineBatchExecutors;

public class MaxPaddedTokensBatchExecutorBuilder<TToken, TOutput>
    : TokenPipelineBatchExecutorBuilder<TToken, TOutput, MaxPaddedTokensBatchExecutorBuilder<TToken, TOutput>>
    where TToken : ITokenizable
{
    public int MaxTokensCount { get; set; }
    public double MaxPaddedRatio { get; set; }

    public override async ValueTask<IPipelineBatchExecutor<TToken, TOutput>> BuildAsync()
    {
        IPipelineBatchExecutor<TToken, TOutput> executor = await CreateInternalPipelineBatchExecutorAsync();
        Console.WriteLine("Using TokenBatchSize chunking and Max Padding");
        executor = new MaxPaddedTokensBatchExecutor<TToken, TOutput>(executor, MaxPaddedRatio, MaxTokensCount);
        Console.WriteLine("Using Sort by token count execution");
        executor = new TokenCountSortingBatchExecutor<TToken, TOutput>(executor, await GetTokenizer());

        return executor;
    }
}