namespace FAI.Core.Configurations.PipelineBatchExecutors;

public sealed record ParallelPipelineExecutorOptions(int BatchSize, int? MaxConcurrency = null)
{
    public ParallelPipelineExecutorOptions() : this(0) { }
}
