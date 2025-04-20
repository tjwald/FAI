namespace ML.Infra.Configurations.PipelineBatchExecutors;

/// <summary>
/// Represents the type of pipeline executor used for processing batches.
/// </summary>
public enum PipelineExecutorType
{
    /// <summary>
    /// Executes tasks sequentially, one after the other.
    /// </summary>
    Serial,

    /// <summary>
    /// Executes tasks in parallel, allowing multiple tasks to run concurrently.
    /// </summary>
    Parallel,

    /// <summary>
    /// Executes tasks in a streamed manner, processing data as it becomes available, and manageing different parts of the task in different in-mem workers.
    /// </summary>
    Streamed,
}
