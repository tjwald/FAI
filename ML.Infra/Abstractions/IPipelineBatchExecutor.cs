namespace ML.Infra.Abstractions;

public interface IPipelineBatchExecutor<TInput, TOutput>
{
    Task ExecuteBatchPredict(IPipeline<TInput, TOutput> pipeline, ReadOnlyMemory<TInput> inputs, Memory<TOutput> outputSpan);
}