namespace FAI.Core.Configurations.PipelineBatchExecutors;

public sealed record StreamedPipelineExecutorOptions(int? BatchSize, int? MaxConcurrency = null, bool ParallelPreProcessing = true)
{
    public StreamedPipelineExecutorOptions() : this(null, null, false)
    {

    }
}
