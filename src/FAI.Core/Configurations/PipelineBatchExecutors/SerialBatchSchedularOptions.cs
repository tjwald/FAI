namespace FAI.Core.Configurations.PipelineBatchExecutors;

public sealed record SerialBatchSchedularOptions(int? BatchSize)
{
    public SerialBatchSchedularOptions() : this((int?)null) { }
}
