namespace ML.Infra.Abstractions;

public interface IInference<TInput, TOutput>
{
    Task<TOutput> Predict(TInput input);
    Task<TOutput[]> BatchPredict(TInput[] input);
}
