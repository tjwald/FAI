namespace ML.Infra.Configurations.PipelineBatchExecutors;

/// <summary>
/// Represents the base interface for pipeline batch executor options.
/// </summary>
public interface IPipelineBatchExecutorOptions;

/// <summary>
/// Represents options for a decorator executor, which wraps another executor.
/// </summary>
/// <param name="InternalExecutorOptions">The internal executor options being decorated.</param>
public record DecoratorExecutorOptions(IPipelineBatchExecutorOptions InternalExecutorOptions) : IPipelineBatchExecutorOptions;

/// <summary>
/// Represents options for a parallel pipeline executor.
/// </summary>
/// <param name="BatchSize">The size of each batch to process.</param>
/// <param name="MaxConcurrency">The maximum number of concurrent tasks allowed. Null for no limit.</param>
public record ParallelPipelineExecutorOptions(int BatchSize, int? MaxConcurrency) : IPipelineBatchExecutorOptions;

/// <summary>
/// Represents options for a serial pipeline executor.
/// </summary>
/// <param name="BatchSize">The size of each batch to process.</param>
public record SerialPipelineExecutorOptions(int BatchSize) : IPipelineBatchExecutorOptions;

/// <summary>
/// Represents options for a streamed pipeline executor.
/// </summary>
/// <param name="BatchSize">The size of each batch to process.</param>
/// <param name="MaxConcurrency">The maximum number of concurrent tasks allowed. Null for no limit.</param>
/// <param name="ParallelPreProcessing">Indicates whether preprocessing should be done in parallel.</param>
public record StreamedPipelineExecutorOptions(int BatchSize, int? MaxConcurrency, bool ParallelPreProcessing) : IPipelineBatchExecutorOptions;
