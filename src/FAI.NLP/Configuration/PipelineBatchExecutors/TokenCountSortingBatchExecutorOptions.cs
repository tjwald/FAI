namespace FAI.NLP.Configuration.PipelineBatchExecutors;

public sealed record TokenCountSortingBatchExecutorOptions(bool Ascending)
{
    public TokenCountSortingBatchExecutorOptions() : this(true) { }
}
