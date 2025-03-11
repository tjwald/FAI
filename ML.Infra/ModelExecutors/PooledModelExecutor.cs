using System.Numerics.Tensors;
using ML.Infra.Abstractions;

namespace ML.Infra.ModelExecutors;

public interface IObjectPool<out T>
{
    T Get();
}

public sealed class PooledModelExecutor<TInput, TOutput>: IModelExecutor<TInput, TOutput>
{
    private readonly IObjectPool<IModelExecutor<TInput, TOutput>> _executorPool;

    public PooledModelExecutor(IObjectPool<IModelExecutor<TInput, TOutput>> executorPool)
    {
        _executorPool = executorPool;
    }

    public async Task<Tensor<TOutput>[]> RunAsync(Tensor<TInput>[] inputs)
    {
        IModelExecutor<TInput, TOutput> executor = _executorPool.Get();
        return await executor.RunAsync(inputs);
    }

    public Task RunAsync(Tensor<TInput>[] inputs, Action<ReadOnlyTensorSpan<TOutput>, int> postProcess)
    {
        IModelExecutor<TInput, TOutput> executor = _executorPool.Get();
        return executor.RunAsync(inputs, postProcess);
    }
}