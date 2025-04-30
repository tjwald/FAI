using ML.Infra.Abstractions;
using ML.Infra.PipelineBatchExecutors;

namespace ML.Infra.Configurations.PipelineBatchExecutors;

public class ParallelPipelineExecutorBuilder<TInput, TOutput> : InferenceStepsPipelineExecutorBuilder<TInput, TOutput, ParallelPipelineExecutorBuilder<TInput, TOutput>>
{
    public int BatchSize { get; set; }
    public int? MaxConcurrency { get; set; } = null;

    public override async ValueTask<IPipelineBatchExecutor<TInput, TOutput>> BuildAsync()
    {
        return new ParallelPipelineBatchExecutor<TInput, TOutput>(await CreateInferenceSteps(), BatchSize, MaxConcurrency);
    }
}