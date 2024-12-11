namespace ML.Infra.Abstractions;

public interface IPipeline<TInput, TOutput>: IInference<TInput, TOutput>
{
    Task ProcessBatch(ReadOnlyMemory<TInput> inputs, Memory<TOutput> outputs);
}

public interface IPipelineBatchExecutor<TInput, TOutput>
{
    Task ExecuteBatchPredict(IPipeline<TInput, TOutput> pipeline, ReadOnlyMemory<TInput> inputs, Memory<TOutput> outputSpan);
}