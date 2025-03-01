using ML.Infra.Abstractions;
using ML.Infra.Configurations.PipelineBatchExecutors;
using ML.Infra.PipelineBatchExecutors;

namespace ML.Infra.Factories;

public static class PipelineBatchExecutorFactory
{
    public static IPipelineBatchExecutor<TInput, TOutput> CreatePipelineBatchExecutor<TInput, TPreprocess, TModelOutput, TOutput>(
        IPipelineBatchExecutorOptions options, IInferenceSteps<TInput, TOutput> inferenceSteps)
    {
        Console.WriteLine($"Using Model Pipeline Executor: {options}");
        IPipelineBatchExecutor<TInput, TOutput> executor = options switch
        {
            SerialPipelineExecutorOptions opt => new SerialPipelineBatchExecutor<TInput, TOutput>(inferenceSteps, maxBatchSize: opt.BatchSize),
            ParallelPipelineExecutorOptions opt => new ParallelPipelineBatchExecutor<TInput, TOutput>(inferenceSteps, opt.BatchSize, opt.MaxConcurrency),
            StreamedPipelineExecutorOptions opt => new StreamedBatchExecutor<TInput, TPreprocess, TModelOutput, TOutput>(
                inferenceSteps, opt.BatchSize, opt.MaxConcurrency, opt.ParallelPreProcessing),
            _ => throw new ArgumentException("Unsupported pipeline executor type")
        };
        return executor;
    }
}