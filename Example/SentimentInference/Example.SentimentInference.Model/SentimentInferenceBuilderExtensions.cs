using System.Diagnostics.CodeAnalysis;
using FAI.Core.Abstractions;
using FAI.Core.Extensions.DI;
using FAI.Core.ResultTypes;
using FAI.NLP.Tokenization;
using Microsoft.Extensions.DependencyInjection;

namespace Example.SentimentInference.Model;


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

public class SentimentInferenceBuilder : PipelineBuilder<TokenizedText, ClassificationResult<bool, float>, SentimentInferenceBuilder>
{
    public SentimentInferenceBuilder(IServiceCollection globalServices, string? key = null) : base(globalServices, key)
    {
    }

    public SentimentInferenceBuilder AddTokenizer(Func<IServiceProvider, PretrainedTokenizer> tokenizerFactory)
    {
        return AddLocal(tokenizerFactory);
    }

    public override IServiceCollection Build()
    {
        base.Build();

        _services.AddSingleton<SentimentInference>();

        if (_key is null)
        {
            _globalServices.AddSingleton(BuildSentimentInference);
        }
        else
        {
            _globalServices.AddKeyedSingleton(_key, BuildSentimentInference);
        }

        return _globalServices;
    }

    private IInference<string, bool> BuildSentimentInference(IServiceProvider globalServiceProvider)
    {
        foreach (ServiceDescriptor serviceDescriptor in _globalServices)
        {
            _services.Add(serviceDescriptor);
        }
        var localServiceProvider = _services.BuildServiceProvider();
        return ActivatorUtilities.CreateInstance<SentimentInference>(localServiceProvider);
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
