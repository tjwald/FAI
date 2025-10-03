namespace FAI.NLP.Configuration.PipelineBatchExecutors;

public sealed record MaxPaddedTokensBatchExecutorOptions(double MaxPaddedTokenRatio, int MaxTokenCount)
{
    public MaxPaddedTokensBatchExecutorOptions() : this(0, 0) { }
}
