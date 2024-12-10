namespace ML.Infra.Abstractions;

public interface IPipelineBatchExecutor<TInput, TOutput, TPreprocess, TModelOutput>
{
    Task ExecuteBatchPredict(Pipeline<TInput, TOutput, TPreprocess, TModelOutput> pipeline, ReadOnlyMemory<TInput> inputs, Memory<TOutput> outputSpan);
}