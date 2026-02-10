using FAI.Core.Abstractions;
using FAI.Core.PipelineBatchExecutors;
using FAI.Core.Pipelines;

namespace FAI.Core.Extensions.DI;

public class PipelineBuilder<TInput, TOutput>
{
    // A list of factories. Each factory takes:
    // 1. 'next': The executor that comes AFTER this one.
    // 2. 'sp': The IServiceProvider to resolve dependencies.
    // It returns: The configured middleware instance.
    private readonly List<Func<IPipelineBatchExecutor<TInput, TOutput>, IServiceProvider, IPipelineBatchExecutor<TInput, TOutput>>> _batchExecutorFactories = new();
    // _sinkFactory is the last pipeline executor to run - it doesn't get a "next" parameter
    private Func<IServiceProvider, IPipelineBatchExecutor<TInput, TOutput>>? _sinkFactory;
    private Func<IServiceProvider, IPipelineBatchExecutor<TInput, TOutput>, IPipeline<TInput, TOutput>>? _pipelineFactory;

    public PipelineBuilder(IServiceCollection serviceCollection)
    {
        ServiceCollection = serviceCollection;
    }

    internal IServiceCollection ServiceCollection { get; }

    public PipelineBuilder<TInput, TOutput> AddServices(Action<IServiceCollection> action)
    {
        action(ServiceCollection);
        return this;
    }

    public PipelineBuilder<TInput, TOutput> AddModelExecutor<TModelInput, TModelOutput>(Func<IServiceProvider, IModelExecutor<TModelInput, TModelOutput>> func)
    {
        ServiceCollection.AddSingleton(func);
        return this;
    }

    public PipelineBuilder<TInput, TOutput> AddInferenceSteps<TInferenceSteps>()
        where TInferenceSteps : class, IInferenceSteps<TInput, TOutput>
    {
        ServiceCollection.AddSingleton<IInferenceSteps<TInput, TOutput>, TInferenceSteps>();
        return this;
    }

    public PipelineBuilder<TInput, TOutput> UsePipeline<TPipeline>(Func<IServiceProvider, IPipelineBatchExecutor<TInput, TOutput>, IPipeline<TInput, TOutput>> factory) where TPipeline : IPipeline<TInput, TOutput>
    {
        _pipelineFactory = factory;
        return this;
    }

    public PipelineBuilder<TInput, TOutput> Use<TBatchExecutor>() where TBatchExecutor : IPipelineBatchExecutor<TInput, TOutput>
    {
        return this.Use<TBatchExecutor>((next, sp) => ActivatorUtilities.CreateInstance<TBatchExecutor>(sp, next));
    }

    public PipelineBuilder<TInput, TOutput> Use<TBatchExecutor>(Func<IPipelineBatchExecutor<TInput, TOutput>, IServiceProvider, IPipelineBatchExecutor<TInput, TOutput>> factory) where TBatchExecutor : IPipelineBatchExecutor<TInput, TOutput>
    {
        _batchExecutorFactories.Add(factory);
        return this;
    }

    public PipelineBuilder<TInput, TOutput> UseSink<TBatchExecutor>(Func<IServiceProvider, IPipelineBatchExecutor<TInput, TOutput>> factory) where TBatchExecutor : IPipelineBatchExecutor<TInput, TOutput>
    {
        _sinkFactory = factory;
        return this;
    }


    internal IPipeline<TInput, TOutput> Build(IServiceProvider sp)
    {
        IPipelineBatchExecutor<TInput, TOutput> current = _sinkFactory is not null ? _sinkFactory(sp) : ActivatorUtilities.CreateInstance<SinkPipelineBatchExecutor<TInput, TOutput>>(sp);
        for (int i = _batchExecutorFactories.Count - 1; i >= 0; i--)
        {
            var batchExecutorFactory = _batchExecutorFactories[i];
            current = batchExecutorFactory(current, sp);
        }

        var pipeline = _pipelineFactory is not null ? _pipelineFactory(sp, current) : new Pipeline<TInput, TOutput>(current);

        return pipeline;
    }
}
