using FAI.Core.Abstractions;
using FAI.Core.PipelineBatchExecutors;

namespace FAI.Core.Configurations.PipelineBatchExecutors;

public class BackgroundPipelineBatchExecutorBuilder<TInput, TOutput> : DecoratorExecutorBuilder<TInput, TOutput,
    BackgroundPipelineBatchExecutorBuilder<TInput, TOutput>>
{
    public int MaxConcurrency { get; set; }

    public override async ValueTask<IPipelineBatchExecutor<TInput, TOutput>> BuildAsync()
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(MaxConcurrency, nameof(MaxConcurrency));

        var internalPipelineExecutor = await CreateInternalPipelineBatchExecutorAsync();
        return new BackgroundPipelineBatchExecutor<TInput, TOutput>(internalPipelineExecutor, MaxConcurrency);
    }
}
