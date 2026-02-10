namespace FAI.Core.Configurations.PipelineBatchExecutors;

public sealed record ParallelBatchSchedularOptions(int? MaxConcurrency)
{
    public ParallelBatchSchedularOptions() : this((int?)null) { }
}
