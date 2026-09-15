using System.Numerics.Tensors;
using FAI.Core.Abstractions;
using FAI.Core.Pipelines;

namespace FAI.Core;

/// <summary>
/// Provides a generic application-level inference facade for pipelines that produce a 2D tensor of feature vectors.
/// </summary>
/// <typeparam name="TInput">The input type.</typeparam>
/// <typeparam name="TElement">The tensor element type.</typeparam>
public class TensorBatchInference<TInput, TElement> : ITensorInference<TInput, TElement>
{
    private readonly IPipeline<ReadOnlyMemory<TInput>, Tensor<TElement>> _pipeline;
    private int? _secondaryDimension;

    /// <summary>
    /// Initializes a new instance of the <see cref="TensorBatchInference{TInput, TElement}"/> class.
    /// </summary>
    /// <param name="pipeline">The underlying pipeline producing tensor outputs.</param>
    /// <param name="secondaryDimension">Optional expected secondary (feature) dimension.</param>
    public TensorBatchInference(
        IPipeline<ReadOnlyMemory<TInput>, Tensor<TElement>> pipeline,
        int? secondaryDimension = null)
    {
        _pipeline = pipeline;
        _secondaryDimension = secondaryDimension;
    }

    /// <summary>
    /// Predicts a feature tensor for one input.
    /// </summary>
    /// <param name="input">The input item to evaluate.</param>
    /// <returns>A 2D tensor with a single row containing the feature vector.</returns>
    public Task<Tensor<TElement>> Predict(TInput input)
        => BatchPredict((TInput[])[input]);

    /// <summary>
    /// Predicts feature vectors for a batch of inputs into a newly allocated or destination tensor.
    /// </summary>
    /// <param name="input">The batch of inputs to evaluate.</param>
    /// <returns>A 2D tensor containing the feature vectors for all inputs.</returns>
    public async Task<Tensor<TElement>> BatchPredict(ReadOnlyMemory<TInput> input)
    {
        if (input.IsEmpty)
        {
            throw new ArgumentException("Cannot execute an empty batch.", nameof(input));
        }

        if (_pipeline is IDestinationPipeline<ReadOnlyMemory<TInput>, Tensor<TElement>> destinationPipeline)
        {
            if (_secondaryDimension is int dimension)
            {
                Tensor<TElement> output = Tensor.CreateFromShape<TElement>([input.Length, dimension]);
                await destinationPipeline.ExecuteAsync(input, output);
                return output;
            }

            Tensor<TElement> result = await _pipeline.ExecuteAsync(input);
            if (result.Lengths.Length >= 2)
            {
                _secondaryDimension = checked((int)result.Lengths[1]);
            }

            return result;
        }

        return await _pipeline.ExecuteAsync(input);
    }

    /// <summary>
    /// Predicts feature vectors for a batch of inputs directly into a caller-supplied destination tensor.
    /// </summary>
    /// <param name="input">The batch of inputs to evaluate.</param>
    /// <param name="destination">The destination tensor to write results into.</param>
    public async Task BatchPredict(ReadOnlyMemory<TInput> input, Tensor<TElement> destination)
    {
        if (input.IsEmpty)
        {
            throw new ArgumentException("Cannot execute an empty batch.", nameof(input));
        }

        if (_pipeline is IDestinationPipeline<ReadOnlyMemory<TInput>, Tensor<TElement>> destinationPipeline)
        {
            await destinationPipeline.ExecuteAsync(input, destination);
            return;
        }

        Tensor<TElement> result = await _pipeline.ExecuteAsync(input);
        result.AsReadOnlyTensorSpan().CopyTo(destination.AsTensorSpan());
    }
}
