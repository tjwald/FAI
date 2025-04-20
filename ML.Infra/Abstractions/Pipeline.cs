namespace ML.Infra.Abstractions;

/// <summary>
/// Represents a machine learning pipeline that processes input data and produces output data.
/// This class supports both single prediction and batch prediction operations.
/// </summary>
/// <typeparam name="TInput">The type of the input data for the pipeline.</typeparam>
/// <typeparam name="TOutput">The type of the output data for the pipeline.</typeparam>
public class Pipeline<TInput, TOutput> : IPipeline<TInput, TOutput>
{
    private readonly IPipelineBatchExecutor<TInput, TOutput> _executor;

    /// <summary>
    /// Initializes a new instance of the <see cref="Pipeline{TInput, TOutput}"/> class.
    /// </summary>
    /// <param name="executor">The batch executor responsible for processing predictions.</param>
    public Pipeline(IPipelineBatchExecutor<TInput, TOutput> executor)
    {
        _executor = executor;
    }

    /// <summary>
    /// Performs a single prediction using the pipeline.
    /// </summary>
    /// <param name="input">The input data for the prediction.</param>
    /// <returns>A task that represents the asynchronous operation, containing the prediction result.</returns>
    public async Task<TOutput> Predict(TInput input)
    {
        TInput[] inputArr = [input];
        TOutput[] outputArr = await BatchPredict(inputArr);
        return outputArr[0];
    }

    /// <summary>
    /// Performs a batch prediction using the pipeline.
    /// </summary>
    /// <param name="inputs">The input data for the batch prediction, provided as a read-only memory block.</param>
    /// <returns>A task that represents the asynchronous operation, containing the prediction results as an array.</returns>
    public async Task<TOutput[]> BatchPredict(ReadOnlyMemory<TInput> inputs)
    {
        var outputs = new TOutput[inputs.Length];
        Memory<TOutput> outputSpan = outputs.AsMemory();

        await _executor.ExecuteBatchPredict(inputs, outputSpan);

        return outputs;
    }
}
