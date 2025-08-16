using System.Numerics.Tensors;
using FAI.Core.Abstractions;

namespace FAI.Core.ModelExecutors;

/// <summary>
/// Represents a pool of reusable objects of type <typeparamref name="T"/>.
/// </summary>
/// <typeparam name="T">The type of objects managed by the pool.</typeparam>
public interface IObjectPool<out T>
{
    /// <summary>
    /// Retrieves an object from the pool.
    /// </summary>
    /// <returns>An instance of type <typeparamref name="T"/> from the pool.</returns>
    T Get();
}

/// <summary>
/// A pooled executor for machine learning models that reuses instances of <see cref="IModelExecutor{TInput,TOutput}"/>.
/// </summary>
/// <typeparam name="TInput">The type of the input data for the model.</typeparam>
/// <typeparam name="TOutput">The type of the output data from the model.</typeparam>
public sealed class PooledModelExecutor<TInput, TOutput> : IModelExecutor<TInput, TOutput>
{
    private readonly IObjectPool<IModelExecutor<TInput, TOutput>> _executorPool;

    /// <summary>
    /// Initializes a new instance of the <see cref="PooledModelExecutor{TInput, TOutput}"/> class.
    /// </summary>
    /// <param name="executorPool">The pool of <see cref="IModelExecutor{TInput, TOutput}"/> instances to be reused.</param>
    public PooledModelExecutor(IObjectPool<IModelExecutor<TInput, TOutput>> executorPool)
    {
        _executorPool = executorPool;
    }

    /// <summary>
    /// Executes the model asynchronously with the given inputs and returns the outputs as an array of tensors.
    /// </summary>
    /// <param name="inputs">An array of input tensors to be processed by the model.</param>
    /// <returns>A task that represents the asynchronous operation. The task result contains an array of output tensors.</returns>
    public async Task<Tensor<TOutput>[]> RunAsync(Tensor<TInput>[] inputs)
    {
        IModelExecutor<TInput, TOutput> executor = _executorPool.Get();
        return await executor.RunAsync(inputs);
    }

    /// <summary>
    /// Executes the model asynchronously with the given inputs and processes the outputs using a provided callback function.
    /// </summary>
    /// <param name="inputs">An array of input tensors to be processed by the model.</param>
    /// <param name="postProcess">
    /// A callback function to process the output tensors. The function receives a read-only span of the output tensor data
    /// and the index of the corresponding input tensor.
    /// </param>
    /// <returns>A task that represents the asynchronous operation.</returns>
    public Task RunAsync(Tensor<TInput>[] inputs, Action<ReadOnlyTensorSpan<TOutput>, int> postProcess)
    {
        IModelExecutor<TInput, TOutput> executor = _executorPool.Get();
        return executor.RunAsync(inputs, postProcess);
    }
}
