using ML.Infra.Abstractions;
using ML.Infra.PipelineBatchExecutors;

namespace ML.Infra.Configurations.PipelineBatchExecutors;

public class SerialPipelineExecutorBuilder<TInput, TOutput> 
    : InferenceStepsPipelineExecutorBuilder<TInput, TOutput, SerialPipelineExecutorBuilder<TInput, TOutput>>
{
    public int? BatchSize { get; set; } = null;

    public override async ValueTask<IPipelineBatchExecutor<TInput, TOutput>> BuildAsync()
    {
        return new SerialPipelineBatchExecutor<TInput, TOutput>(await CreateInferenceSteps(), BatchSize);
    }
}