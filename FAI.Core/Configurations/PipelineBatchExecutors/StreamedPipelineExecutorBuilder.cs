using FAI.Core.Abstractions;
using FAI.Core.PipelineBatchExecutors;

namespace FAI.Core.Configurations.PipelineBatchExecutors;

public class StreamedPipelineExecutorBuilder<TInput, TPreprocess, TModelOutput, TOutput> : IPipelineBatchExecutorBuilder<TInput, TOutput>
{
    public int? BatchSize { get; set; }
    public int? MaxConcurrency { get; set; } = null;
    public bool ParallelPreProcessing { get; set; } = true;

    private Func<ValueTask<InferenceSteps<TInput, TPreprocess, TModelOutput, TOutput>>>? _inferenceStepsFactory;
    private InferenceSteps<TInput, TPreprocess, TModelOutput, TOutput>? _inferenceSteps;

    private async ValueTask<InferenceSteps<TInput, TPreprocess, TModelOutput, TOutput>> CreateInferenceSteps() =>
        _inferenceSteps ??= await _inferenceStepsFactory!();

    public StreamedPipelineExecutorBuilder<TInput, TPreprocess, TModelOutput, TOutput> UseInferenceSteps<TBuilder, TInferenceSteps>(
        Action<TBuilder> inferenceStepsFactory)
        where TInferenceSteps : InferenceSteps<TInput, TPreprocess, TModelOutput, TOutput>
        where TBuilder : IInferenceStepsBuilder<TInput, TOutput, TInferenceSteps>, new()
    {
        _inferenceStepsFactory = async () =>
        {
            var builder = new TBuilder();
            inferenceStepsFactory(builder);
            return await builder.BuildAsync();
        };
        return this;
    }

    public async ValueTask<IPipelineBatchExecutor<TInput, TOutput>> BuildAsync()
    {
        return new StreamedBatchExecutor<TInput, TPreprocess, TModelOutput, TOutput>(
            await CreateInferenceSteps(), BatchSize, MaxConcurrency, ParallelPreProcessing);
    }
}
