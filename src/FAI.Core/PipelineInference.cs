using FAI.Core.Abstractions;
using FAI.Core.Pipelines;

namespace FAI.Core;

/// <summary>
/// A generic application-level inference implementation that delegates to an underlying pipeline.
/// </summary>
/// <typeparam name="TInput">The input type.</typeparam>
/// <typeparam name="TOutput">The output type.</typeparam>
public class PipelineInference<TInput, TOutput> : IInference<TInput, TOutput>
{
    private readonly IPipeline<TInput, TOutput> _pipeline;

    /// <summary>
    /// Initializes a new instance of the <see cref="PipelineInference{TInput, TOutput}"/> class.
    /// </summary>
    /// <param name="pipeline">The underlying pipeline to execute.</param>
    public PipelineInference(IPipeline<TInput, TOutput> pipeline)
    {
        ArgumentNullException.ThrowIfNull(pipeline);
        _pipeline = pipeline;
    }

    /// <summary>
    /// Predicts output for the given input using the underlying pipeline.
    /// </summary>
    /// <param name="input">The input to evaluate.</param>
    /// <returns>The predicted output.</returns>
    public async Task<TOutput> Predict(TInput input)
    {
        return await _pipeline.ExecuteAsync(input);
    }

    /// <summary>
    /// Predicts output into a caller-provided destination if supported by the underlying pipeline.
    /// </summary>
    /// <param name="input">The input to evaluate.</param>
    /// <param name="destination">The destination buffer to write results into.</param>
    public async Task Predict(TInput input, TOutput destination)
    {
        if (_pipeline is IDestinationPipeline<TInput, TOutput> destinationPipeline)
        {
            await destinationPipeline.ExecuteAsync(input, destination);
            return;
        }

        throw new NotSupportedException(
            $"The underlying pipeline '{_pipeline.GetType().Name}' does not implement IDestinationPipeline<{typeof(TInput).Name}, {typeof(TOutput).Name}>.");
    }
}
