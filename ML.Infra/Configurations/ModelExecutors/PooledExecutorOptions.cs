namespace ML.Infra.Configurations.ModelExecutors;

public record PooledExecutorOptions<TConfig>(TConfig ExecutorConfig, int ExecutorCount) : IModelExecutorConfig;