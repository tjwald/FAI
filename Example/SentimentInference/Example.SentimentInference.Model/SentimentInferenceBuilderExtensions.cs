using System.Diagnostics.CodeAnalysis;
using FAI.Core.Abstractions;
using FAI.Core.Extensions.DI;
using FAI.Core.ResultTypes;
using FAI.NLP.Tokenization;
using Microsoft.Extensions.DependencyInjection;

namespace Example.SentimentInference.Model;


public static class SentimentInferenceBuilderExtensions
{
    public static PipelineBuilder<TokenizedText, ClassificationResult<bool, float>> AddSentimentInference(this IServiceCollection serviceCollection)
    {
        serviceCollection.AddSingleton<IInference<string, bool>, SentimentInference>();

        return new PipelineBuilder<TokenizedText, ClassificationResult<bool, float>>(serviceCollection);
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
