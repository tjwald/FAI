using System.Diagnostics.CodeAnalysis;
using FAI.Core.Abstractions;
using FAI.Core.ResultTypes;
using FAI.NLP.Tokenization;
using Microsoft.Extensions.DependencyInjection;

namespace Example.SentimentInference.Model;

using PipelineBatchExecutor = IPipelineBatchExecutor<TokenizedText, ClassificationResult<bool, float>>;

public static class SentimentInferenceBuilderExtensions
{
    public static SentimentInferenceBuilder AddSentimentInference(this IServiceCollection services)
    {
        return new SentimentInferenceBuilder(services);
    }

    public static SentimentInferenceBuilder AddKeyedSentimentInference(this IServiceCollection services, string key)
    {
        return new SentimentInferenceBuilder(services, key);
    }
}

public class SentimentInferenceBuilder
{
    private readonly IServiceCollection _builder;
    private readonly IServiceCollection _services;
    private readonly string? _key;
    private bool _isBuilt;

    public SentimentInferenceBuilder(IServiceCollection builder, string? key = null)
    {
        _builder = builder;
        _key = key;
        _services = new ServiceCollection();
    }

    public SentimentInferenceBuilder AddLocal<TService, TImplementation>(TImplementation implementation)
        where TService : class where TImplementation : class, TService
    {
        _services.AddSingleton<TService>(implementation);
        return this;
    }

    public SentimentInferenceBuilder AddTokenizer(Func<IServiceProvider, PretrainedTokenizer> tokenizerFactory)
    {
        _services.AddSingleton(tokenizerFactory);
        return this;
    }


    public SentimentInferenceBuilder AddModelExecutor<TModelInput, TModelOutput>(Func<IServiceProvider, IModelExecutor<TModelInput, TModelOutput>> func)
    {
        _services.AddSingleton(func);
        return this;
    }

    public SentimentInferenceBuilder AddInferenceSteps<TInferenceSteps>()
        where TInferenceSteps : class, IInferenceSteps<TokenizedText, ClassificationResult<bool, float>>
    {
        _services.AddSingleton<IInferenceSteps<TokenizedText, ClassificationResult<bool, float>>, TInferenceSteps>();
        return this;
    }

    public SentimentInferenceBuilder AddBatchExecutor<TBatchExecutor>() where TBatchExecutor : class, PipelineBatchExecutor
    {
        _services.AddSingleton<PipelineBatchExecutor, TBatchExecutor>();
        return this;
    }

    public SentimentInferenceBuilder AddBatchExecutor<TBatchExecutor>(
        Func<IServiceProvider, TBatchExecutor> batchExecutorFactory) where TBatchExecutor : class, PipelineBatchExecutor
    {
        _services.AddSingleton(batchExecutorFactory);
        return this;
    }

    public SentimentInferenceBuilder AddBatchExecutor(Action<IServiceCollection> batchExecutorFactory)
    {
        batchExecutorFactory(_services);
        return this;
    }

    public IServiceCollection Build()
    {
        _isBuilt = _isBuilt ? throw new InvalidOperationException("This builder has already been built.") : true;

        _services.AddSingleton<IPipeline<TokenizedText, ClassificationResult<bool, float>>, Pipeline<TokenizedText, ClassificationResult<bool, float>>>();
        _services.AddSingleton<SentimentInference>();

        if (_key is null)
        {
            _builder.AddSingleton(BuildSentimentInference);
        }
        else
        {
            _services.AddKeyedSingleton(_key, BuildSentimentInference);
        }

        return _builder;
    }

    private IInference<string, bool> BuildSentimentInference(IServiceProvider globalServiceProvider)
    {
        foreach (ServiceDescriptor serviceDescriptor in _builder)
        {
            _services.Add(serviceDescriptor);
        }
        var localServiceProvider = _services.BuildServiceProvider();
        return ActivatorUtilities.CreateInstance<SentimentInference>(localServiceProvider);
    }
}


public class HybridServiceProvider : IServiceProvider
{
    private readonly IServiceProvider _local;
    private readonly IServiceProvider _global;

    public HybridServiceProvider(IServiceProvider local, IServiceProvider global)
    {
        _local = local;
        _global = global;
    }

    public object? GetService(Type serviceType)
    {
        return _local.GetService(serviceType) ?? _global.GetService(serviceType);
    }
}

public class DecoratorChainBuilder
{
    private readonly IServiceCollection _services;
    private Type? _currentType;

    public DecoratorChainBuilder(IServiceCollection services)
    {
        _services = services;
    }

    public DecoratorChainBuilder AddInitial<T>() where T : class
    {
        if (_currentType is not null) throw new InvalidOperationException("The initial type has already been selected");
        _services.AddSingleton<T>();
        _currentType = typeof(T);
        return this;
    }

    public DecoratorChainBuilder Decorate<TDecorator>() where TDecorator : class
    {
        GuardTypeInitialized();
        var currentType = this._currentType;
        _services.AddSingleton<TDecorator>(sp =>
        {
            var decorated = sp.GetRequiredService(currentType);
            return ActivatorUtilities.CreateInstance<TDecorator>(sp, decorated);
        });
        this._currentType = typeof(TDecorator);
        return this;
    }

    public DecoratorChainBuilder Build<TService>() where TService : class
    {
        GuardTypeInitialized();
        if (!_currentType.IsAssignableTo(typeof(TService)))
        {
            throw new InvalidOperationException($"The decorator chain can't be built since last type: \"{_currentType.FullName}\" doesn't match build type \"{typeof(TService).FullName}\"");
        }
        var currentType = this._currentType;

        _services.AddSingleton<TService>(sp => (TService)sp.GetRequiredService(currentType));
        return this;
    }

    [MemberNotNull(nameof(_currentType))]
    private void GuardTypeInitialized()
    {
        if (_currentType is null) throw new InvalidOperationException("The decorator type has to be selected first");
    }
}
