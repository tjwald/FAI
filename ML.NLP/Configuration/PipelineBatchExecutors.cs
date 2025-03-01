using ML.Infra.Configurations.PipelineBatchExecutors;

namespace ML.NLP.Configuration;

public record TokenBasedBatchExecutorOptions(IPipelineBatchExecutorOptions InternalExecutorOptions, bool SortTokens, int? MaxTokensCount = null): IPipelineBatchExecutorOptions;
