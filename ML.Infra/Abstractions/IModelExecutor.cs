using System.Numerics.Tensors;

namespace ML.Infra.Abstractions;

public interface IModelExecutor<TInput, TOutput>
{
    Task<Tensor<TOutput>[]> RunAsync(Tensor<TInput>[] inputs);
}