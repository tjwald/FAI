using System.Numerics.Tensors;

namespace FAI.Core.Abstractions;

/// <summary>
/// Represents a contract for executing machine learning models with specified input and output types.
/// </summary>
/// <typeparam name="TInput">The type of the input data for the model.</typeparam>
/// <typeparam name="TOutput">The type of the output data from the model.</typeparam>
public interface IModelExecutor<TInput, TOutput>
{
    /// <summary>
    /// Executes the model asynchronously with the given inputs and returns the outputs as an array of tensors.
    /// </summary>
    /// <param name="inputs">An array of input tensors to be processed by the model.</param>
    /// <returns>A task that represents the asynchronous operation. The task result contains an array of output tensors.</returns>
    Task<Tensor<TOutput>[]> RunAsync(Tensor<TInput>[] inputs);

    /// <summary>
    /// Executes the model asynchronously with the given inputs and processes the outputs using a provided callback function.
    /// </summary>
    /// <param name="inputs">An array of input tensors to be processed by the model.</param>
    /// <param name="postProcess">
    /// A callback function to process the output tensors. The function receives a read-only span of the output tensor data
    /// and the index of the corresponding input tensor.
    /// </param>
    /// <returns>A task that represents the asynchronous operation.</returns>
    Task RunAsync(Tensor<TInput>[] inputs, Action<ReadOnlyTensorSpan<TOutput>, int> postProcess);
}