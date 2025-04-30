using ML.Infra.Abstractions;

namespace ML.Infra.Configurations.PipelineBatchExecutors;

public abstract class DecoratorExecutorBuilder<TInput, TOutput, TSelf>
    : IPipelineBatchExecutorBuilder<TInput, TOutput>
    where TSelf : DecoratorExecutorBuilder<TInput, TOutput, TSelf>
{
    private Func<IPipelineBatchExecutorBuilder<TInput, TOutput>>? _createInnerPipelineExecutorBuilder;

    protected async ValueTask<IPipelineBatchExecutor<TInput, TOutput>> CreateInternalPipelineBatchExecutorAsync() =>
        await _createInnerPipelineExecutorBuilder!().BuildAsync();

    public TSelf UseInnerPipelineExecutor<TBuilder>(Action<TBuilder> createInnerPipelineExecutorBuilder)
        where TBuilder : IPipelineBatchExecutorBuilder<TInput, TOutput>, new()
    {
        _createInnerPipelineExecutorBuilder = () =>
        {
            TBuilder builder = new TBuilder();
            createInnerPipelineExecutorBuilder(builder);
            return builder;
        };
        return (TSelf)this;
    }

    public abstract ValueTask<IPipelineBatchExecutor<TInput, TOutput>> BuildAsync();
}