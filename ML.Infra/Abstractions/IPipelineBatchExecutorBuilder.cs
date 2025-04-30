namespace ML.Infra.Abstractions;

public interface IPipelineBatchExecutorBuilder<TInput, TOutput>
{
    ValueTask<IPipelineBatchExecutor<TInput, TOutput>> BuildAsync();
}