namespace ML.Infra.Abstractions;

public interface IPipelineBatchExecutor<TInput, TOutput>
{
    Task ExecuteBatchPredict(ReadOnlyMemory<TInput> inputs, Memory<TOutput> outputSpan);
}