using ML.Infra.Configurations.PipelineBatchExecutors;

namespace ML.NLP.Configuration;


/// <summary>
/// Represents options for executing batch operations based on token management.
/// </summary>
/// <param name="InternalExecutorOptions">The internal batch executor options that define execution behavior.</param>
/// <param name="SortTokens">Indicates whether tokens should be sorted before execution.</param>
/// <param name="MaxTokensCount">The optional maximum number of tokens allowed in a batch; <c>null</c> means no limit.</param>
public record TokenBasedBatchExecutorOptions(IPipelineBatchExecutorOptions InternalExecutorOptions, bool SortTokens, int? MaxTokensCount = null): DecoratorExecutorOptions(InternalExecutorOptions);


/// <summary>
/// Represents options for executing batch operations with constraints on maximum padded tokens.
/// </summary>
/// <param name="InternalExecutorOptions">The internal batch executor options that define execution behavior.</param>
/// <param name="MaxTokensCount">The maximum number of tokens allowed in a batch.</param>
/// <param name="MaxPaddedRatio">
/// The maximum allowed ratio of padding within the batch, controlling efficiency in token utilization.
/// </param>
public record MaxPaddedTokensBatchExecutorOptions(IPipelineBatchExecutorOptions InternalExecutorOptions, int MaxTokensCount, double MaxPaddedRatio): DecoratorExecutorOptions(InternalExecutorOptions);
