using ML.Infra.Abstractions;
using ML.Infra.Configurations.PipelineBatchExecutors;
using ML.Infra.PipelineBatchExecutors;

namespace ML.Infra.Factories;


/// <summary>
/// Factory class for creating instances of <see cref="IPipelineBatchExecutor{TInput, TOutput}"/>.
/// This factory supports multiple execution strategies, including serial, parallel, and streamed execution.
/// </summary>
public static class PipelineBatchExecutorFactory
{
    /// <summary>
    /// Creates an instance of <see cref="IPipelineBatchExecutor{TInput, TOutput}"/> based on the provided options and inference steps.
    /// </summary>
    /// <typeparam name="TInput">The type of the input data for the pipeline.</typeparam>
    /// <typeparam name="TPreprocess">The type of the preprocessing step output, used in streamed execution.</typeparam>
    /// <typeparam name="TModelOutput">The type of the model output, used in streamed execution.</typeparam>
    /// <typeparam name="TOutput">The type of the final output data for the pipeline.</typeparam>
    /// <param name="options">The options specifying the type of pipeline executor to create.</param>
    /// <param name="inferenceSteps">The inference steps to be used by the pipeline executor.</param>
    /// <returns>An instance of <see cref="IPipelineBatchExecutor{TInput, TOutput}"/> configured according to the provided options.</returns>
    /// <exception cref="ArgumentException">
    /// Thrown when the provided options are unsupported or when the inference steps are incompatible with the selected executor type.
    /// </exception>
    public static IPipelineBatchExecutor<TInput, TOutput> CreatePipelineBatchExecutor<TInput, TPreprocess, TModelOutput, TOutput>(
        IPipelineBatchExecutorOptions options, IInferenceSteps<TInput, TOutput> inferenceSteps)
    {
        Console.WriteLine($"Using Model Pipeline Executor: {options}");
        IPipelineBatchExecutor<TInput, TOutput> executor = options switch
        {
            SerialPipelineExecutorOptions opt => new SerialPipelineBatchExecutor<TInput, TOutput>(inferenceSteps, maxBatchSize: opt.BatchSize),
            ParallelPipelineExecutorOptions opt => new ParallelPipelineBatchExecutor<TInput, TOutput>(inferenceSteps, opt.BatchSize, opt.MaxConcurrency),
            StreamedPipelineExecutorOptions opt when inferenceSteps is InferenceSteps<TInput, TPreprocess, TModelOutput, TOutput> typedSteps
                => new StreamedBatchExecutor<TInput, TPreprocess, TModelOutput, TOutput>(
                    typedSteps, opt.BatchSize, opt.MaxConcurrency, opt.ParallelPreProcessing),
            StreamedPipelineExecutorOptions _ => throw new ArgumentException("The provided inferenceSteps is not compatible with StreamedPipelineExecutorOptions."),
            _ => throw new ArgumentException("Unsupported pipeline executor type")
        };
        return executor;
    }
}
