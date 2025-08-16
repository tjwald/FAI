using FAI.Core.Abstractions;
using FAI.Core.PipelineBatchExecutors;

namespace FAI.Core.Configurations.PipelineBatchExecutors;

public class SerialPipelineExecutorBuilder<TInput, TOutput>
    : InferenceStepsPipelineExecutorBuilder<TInput, TOutput, SerialPipelineExecutorBuilder<TInput, TOutput>>
{
    public int? BatchSize { get; set; } = null;

    public override async ValueTask<IPipelineBatchExecutor<TInput, TOutput>> BuildAsync()
    {
        return new SerialPipelineBatchExecutor<TInput, TOutput>(await CreateInferenceSteps(), BatchSize);
    }
}
