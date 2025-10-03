namespace FAI.NLP.Configuration.PipelineBatchExecutors;

public sealed record TokenBatchSizeBatchExecutorOptions(int MaxTokensCount)
{
    public TokenBatchSizeBatchExecutorOptions() : this(0) { }
}
