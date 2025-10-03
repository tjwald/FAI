namespace FAI.Core.Configurations.PipelineBatchExecutors;

public sealed record SerialPipelineBatchExecutorOptions(int? BatchSize)
{
    public SerialPipelineBatchExecutorOptions() : this((int?)null) { }
}
