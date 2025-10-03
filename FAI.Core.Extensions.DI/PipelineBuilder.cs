using FAI.Core.Abstractions;

namespace FAI.Core.Extensions.DI;

public class PipelineBuilder<TInput, TOutput>
{
    protected readonly IServiceCollection _serviceCollection;
    private bool _isBuilt;

    public PipelineBuilder(IServiceCollection serviceCollection, string? key = null)
    {
        _serviceCollection = serviceCollection;
    }

    public PipelineBuilder<TInput, TOutput> AddModelExecutor<TModelInput, TModelOutput>(Func<IServiceProvider, IModelExecutor<TModelInput, TModelOutput>> func)
    {
        _serviceCollection.AddSingleton(func);
        return this;
    }

    public PipelineBuilder<TInput, TOutput> AddInferenceSteps<TInferenceSteps>()
        where TInferenceSteps : class, IInferenceSteps<TInput, TOutput>
    {
        _serviceCollection.AddSingleton<IInferenceSteps<TInput, TOutput>, TInferenceSteps>();
        return this;
    }

    public PipelineBuilder<TInput, TOutput> AddBatchExecutor<TBatchExecutor>() where TBatchExecutor : class, IPipelineBatchExecutor<TInput, TOutput>
    {
        _serviceCollection.AddSingleton<IPipelineBatchExecutor<TInput, TOutput>, TBatchExecutor>();
        return this;
    }

    public PipelineBuilder<TInput, TOutput> AddBatchExecutor<TBatchExecutor>(
        Func<IServiceProvider, TBatchExecutor> batchExecutorFactory) where TBatchExecutor : class, IPipelineBatchExecutor<TInput, TOutput>
    {
        _serviceCollection.AddSingleton(batchExecutorFactory);
        return this;
    }

    public PipelineBuilder<TInput, TOutput> AddBatchExecutor(Func<IServiceCollection, Func<IServiceProvider, IPipelineBatchExecutor<TInput, TOutput>>> batchExecutorFactory)
    {
        _serviceCollection.AddSingleton(batchExecutorFactory(_serviceCollection));
        return this;
    }

    public IServiceCollection Build()
    {
        _isBuilt = _isBuilt ? throw new InvalidOperationException("This builder has already been built.") : true;

        _serviceCollection.AddSingleton<IPipeline<TInput, TOutput>, Pipeline<TInput, TOutput>>();

        return _serviceCollection;
    }
}
