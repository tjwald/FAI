namespace FAI.Core.Abstractions;

public interface IPipelineBatchExecutorBuilder<TInput, TOutput>
{
    ValueTask<IPipelineBatchExecutor<TInput, TOutput>> BuildAsync();
}