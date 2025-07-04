namespace FAI.Core.Configurations.ModelExecutors;

/// <summary>
/// Represents options for configuring an executors that pooles the internal executor.
/// </summary>
/// <typeparam name="TConfig">The type of the executor configuration.</typeparam>
/// <param name="ExecutorConfig">The configuration for the executor.</param>
/// <param name="ExecutorCount">The number of executors in the pool.</param>
public record PooledExecutorOptions<TConfig>(TConfig ExecutorConfig, int ExecutorCount) : IModelExecutorConfig;