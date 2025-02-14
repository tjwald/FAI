namespace ML.Infra.Abstractions;

public interface IPipeline<TInput, TOutput>: IInference<TInput, TOutput>;