using FAI.Core.Abstractions;
using Microsoft.Extensions.DependencyInjection;

namespace FAI.Core.Extensions.DI;

public class PipelineBuilder<TInput, TOutput, TSelf> where TSelf : PipelineBuilder<TInput, TOutput, TSelf>
{
    protected readonly IServiceCollection _globalServices;
    protected readonly IServiceCollection _services;
    protected readonly string? _key;
    private bool _isBuilt;

    public PipelineBuilder(IServiceCollection globalServices, string? key = null)
    {
        _globalServices = globalServices;
        _key = key;
        _services = new ServiceCollection();
    }

    public TSelf AddLocal<TService>(Func<IServiceProvider, TService> factory)
        where TService : class
    {
        _services.AddSingleton(factory);
        return (TSelf)this;
    }

    public TSelf AddLocal<TService, TImplementation>(TImplementation implementation)
        where TService : class where TImplementation : class, TService
    {
        _services.AddSingleton<TService>(implementation);
        return (TSelf)this;
    }

    public TSelf AddModelExecutor<TModelInput, TModelOutput>(Func<IServiceProvider, IModelExecutor<TModelInput, TModelOutput>> func)
    {
        _services.AddSingleton(func);
        return (TSelf)this;
    }

    public TSelf AddInferenceSteps<TInferenceSteps>()
        where TInferenceSteps : class, IInferenceSteps<TInput, TOutput>
    {
        _services.AddSingleton<IInferenceSteps<TInput, TOutput>, TInferenceSteps>();
        return (TSelf)this;
    }

    public TSelf AddBatchExecutor<TBatchExecutor>() where TBatchExecutor : class, IPipelineBatchExecutor<TInput, TOutput>
    {
        _services.AddSingleton<IPipelineBatchExecutor<TInput, TOutput>, TBatchExecutor>();
        return (TSelf)this;
    }

    public TSelf AddBatchExecutor<TBatchExecutor>(
        Func<IServiceProvider, TBatchExecutor> batchExecutorFactory) where TBatchExecutor : class, IPipelineBatchExecutor<TInput, TOutput>
    {
        _services.AddSingleton(batchExecutorFactory);
        return (TSelf)this;
    }

    public TSelf AddBatchExecutor(Action<IServiceCollection> batchExecutorFactory)
    {
        batchExecutorFactory(_services);
        return (TSelf)this;
    }

    public virtual IServiceCollection Build()
    {
        _isBuilt = _isBuilt ? throw new InvalidOperationException("This builder has already been built.") : true;

        _services.AddSingleton<IPipeline<TInput, TOutput>, Pipeline<TInput, TOutput>>();

        return _globalServices;
    }
}
