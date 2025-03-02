using ML.Infra.Configurations.PipelineBatchExecutors;

namespace ML.NLP.Configuration;


public record TokenBasedBatchExecutorOptions(IPipelineBatchExecutorOptions InternalExecutorOptions, bool SortTokens, int? MaxTokensCount = null): DecoratorExecutorOptions(InternalExecutorOptions);

public record MaxPaddedTokensBatchExecutorOptions(IPipelineBatchExecutorOptions InternalExecutorOptions, int MaxTokensCount, double MaxPaddedRatio): DecoratorExecutorOptions(InternalExecutorOptions);