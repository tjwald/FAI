namespace FAI.Core.Configurations.PipelineBatchExecutors;

public sealed record BackgroundPipelineBatchExecutorOptions(int MaxConcurrency)
{
    internal BackgroundPipelineBatchExecutorOptions() : this(0) { }
}
