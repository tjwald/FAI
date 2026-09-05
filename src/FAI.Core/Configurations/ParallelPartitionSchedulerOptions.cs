namespace FAI.Core.Configurations;

public sealed record ParallelPartitionSchedulerOptions(int? MaxConcurrency)
{
    public ParallelPartitionSchedulerOptions() : this((int?)null) { }
}
