using FAI.Core.Abstractions;

namespace FAI.Core.Configurations.PipelineBatchExecutors;

public abstract class InferenceStepsPipelineExecutorBuilder<TInput, TOutput, TSelf>
    : IPipelineBatchExecutorBuilder<TInput, TOutput>
    where TSelf : InferenceStepsPipelineExecutorBuilder<TInput, TOutput, TSelf>
{
    private Func<ValueTask<IInferenceSteps<TInput, TOutput>>>? _inferenceStepsFactory;
    private IInferenceSteps<TInput, TOutput>? _inferenceSteps;
    protected async ValueTask<IInferenceSteps<TInput, TOutput>> CreateInferenceSteps() => _inferenceSteps ??= await _inferenceStepsFactory!();

    public TSelf UseInferenceSteps<TBuilder, TInferenceSteps>(
        Action<TBuilder> inferenceStepsFactory)
        where TInferenceSteps : IInferenceSteps<TInput, TOutput>
        where TBuilder : IInferenceStepsBuilder<TInput, TOutput, TInferenceSteps>, new()
    {
        _inferenceStepsFactory = async () =>
        {
            var builder = new TBuilder();
            inferenceStepsFactory(builder);
            return await builder.BuildAsync();
        };
        return (TSelf)this;
    }

    public abstract ValueTask<IPipelineBatchExecutor<TInput, TOutput>> BuildAsync();
}
