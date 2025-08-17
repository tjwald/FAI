using FAI.Core.Abstractions;
using FAI.Core.PipelineBatchExecutors;

namespace FAI.Core.Configurations.PipelineBatchExecutors;

public class RoutingPipelineExecutorBuilder<TInput, TOutput> : IPipelineBatchExecutorBuilder<TInput, TOutput>
{
    private readonly List<IPipelineBatchExecutorBuilder<TInput, TOutput>> _pipelineExecutorBuilders = [];
    private IBatchExecutionRoutingStrategy<TInput, TOutput>? _routingStrategy;

    public RoutingPipelineExecutorBuilder<TInput, TOutput> UsePipelineExecutorBuilder<TBuilder>(
        Action<TBuilder> pipelineExecutorBuilderFactory) where TBuilder : IPipelineBatchExecutorBuilder<TInput, TOutput>, new()
    {
        TBuilder builder = new TBuilder();
        pipelineExecutorBuilderFactory(builder);
        _pipelineExecutorBuilders.Add(builder);
        return this;
    }

    public RoutingPipelineExecutorBuilder<TInput, TOutput> UseRoutingStrategy(IBatchExecutionRoutingStrategy<TInput, TOutput> routingStrategy)
    {
        _routingStrategy = routingStrategy;
        return this;
    }

    public async ValueTask<IPipelineBatchExecutor<TInput, TOutput>> BuildAsync()
    {
        var pipelineExecutors = new IPipelineBatchExecutor<TInput, TOutput>[_pipelineExecutorBuilders.Count];
        for (int i = 0; i < _pipelineExecutorBuilders.Count; i++)
        {
            pipelineExecutors[i] = await _pipelineExecutorBuilders[i].BuildAsync();
        }

        return new RoutingPipelineBatchExecutor<TInput, TOutput>(pipelineExecutors, _routingStrategy!);
    }
}
