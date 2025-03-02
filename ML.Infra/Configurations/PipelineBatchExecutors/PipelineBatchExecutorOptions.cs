namespace ML.Infra.Configurations.PipelineBatchExecutors;

public interface IPipelineBatchExecutorOptions;

public record DecoratorExecutorOptions(IPipelineBatchExecutorOptions InternalExecutorOptions) : IPipelineBatchExecutorOptions;

public record ParallelPipelineExecutorOptions(int BatchSize, int? MaxConcurrency) : IPipelineBatchExecutorOptions;

public record SerialPipelineExecutorOptions(int BatchSize) : IPipelineBatchExecutorOptions;

public record StreamedPipelineExecutorOptions(int BatchSize, int? MaxConcurrency, bool ParallelPreProcessing) : IPipelineBatchExecutorOptions;
