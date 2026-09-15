using FAI.Core.Pipelines;

namespace FAI.Core.Extensions.DI;

internal sealed class IdentityPipeline<T> : IPipeline<T, T>
{
    public ValueTask<T> ExecuteAsync(T input, CancellationToken cancellationToken = default) => ValueTask.FromResult(input);
}

internal interface IStage<TStart, TCurrent>
{
    IStage<TStart, TNext> Append<TNext>(Func<IServiceProvider, IPipeline<TCurrent, TNext>> pipelineFactory);

    IStage<TStart, TCurrent> Use(IForwardPipelineDecorator<TCurrent> decorator);

    IPipeline<TStart, TCurrent> Build(IServiceProvider serviceProvider);
}

internal sealed class InitialStage<T> : IStage<T, T>
{
    public IStage<T, TNext> Append<TNext>(Func<IServiceProvider, IPipeline<T, TNext>> pipelineFactory)
        => new ComposedStage<T, TNext>(pipelineFactory);

    public IStage<T, T> Use(IForwardPipelineDecorator<T> decorator)
        => new DecoratedStage<T, T, T>(
            buildPrefix: null,
            buildSuffix: _ => new IdentityPipeline<T>(),
            decorators: [decorator],
            suffixIsEmpty: true);

    public IPipeline<T, T> Build(IServiceProvider serviceProvider)
        => new IdentityPipeline<T>();
}

internal sealed class ComposedStage<TStart, TCurrent> : IStage<TStart, TCurrent>
{
    private readonly Func<IServiceProvider, IPipeline<TStart, TCurrent>> _build;

    public ComposedStage(Func<IServiceProvider, IPipeline<TStart, TCurrent>> build)
    {
        _build = build;
    }

    public IStage<TStart, TNext> Append<TNext>(Func<IServiceProvider, IPipeline<TCurrent, TNext>> pipelineFactory)
        => new ComposedStage<TStart, TNext>(sp => AppendedPipeline.Create(_build(sp), pipelineFactory(sp)));

    public IStage<TStart, TCurrent> Use(IForwardPipelineDecorator<TCurrent> decorator)
        => new DecoratedStage<TStart, TCurrent, TCurrent>(
            _build,
            _ => new IdentityPipeline<TCurrent>(),
            [decorator],
            suffixIsEmpty: true);

    public IPipeline<TStart, TCurrent> Build(IServiceProvider serviceProvider)
        => _build(serviceProvider);
}

internal sealed class DecoratedStage<TStart, TBoundary, TCurrent> : IStage<TStart, TCurrent>
{
    private readonly Func<IServiceProvider, IPipeline<TStart, TBoundary>>? _buildPrefix;
    private readonly Func<IServiceProvider, IPipeline<TBoundary, TCurrent>> _buildSuffix;
    private readonly IReadOnlyList<IForwardPipelineDecorator<TBoundary>> _decorators;
    private readonly bool _suffixIsEmpty;

    public DecoratedStage(
        Func<IServiceProvider, IPipeline<TStart, TBoundary>>? buildPrefix,
        Func<IServiceProvider, IPipeline<TBoundary, TCurrent>> buildSuffix,
        IReadOnlyList<IForwardPipelineDecorator<TBoundary>> decorators,
        bool suffixIsEmpty)
    {
        _buildPrefix = buildPrefix;
        _buildSuffix = buildSuffix;
        _decorators = decorators;
        _suffixIsEmpty = suffixIsEmpty;
    }

    public IStage<TStart, TNext> Append<TNext>(Func<IServiceProvider, IPipeline<TCurrent, TNext>> pipelineFactory)
    {
        Func<IServiceProvider, IPipeline<TBoundary, TNext>> newSuffix = _suffixIsEmpty
            ? sp => (IPipeline<TBoundary, TNext>)(object)pipelineFactory(sp)
            : sp => AppendedPipeline.Create(_buildSuffix(sp), pipelineFactory(sp));

        return new DecoratedStage<TStart, TBoundary, TNext>(_buildPrefix, newSuffix, _decorators, false);
    }

    public IStage<TStart, TCurrent> Use(IForwardPipelineDecorator<TCurrent> decorator)
    {
        if (_suffixIsEmpty)
        {
            var decoratorAtBoundary = (IForwardPipelineDecorator<TBoundary>)(object)decorator;
            return new DecoratedStage<TStart, TBoundary, TCurrent>(
                _buildPrefix,
                _buildSuffix,
                [.. _decorators, decoratorAtBoundary],
                true);
        }

        Func<IServiceProvider, IPipeline<TStart, TCurrent>> currentPipeline = Build;
        return new DecoratedStage<TStart, TCurrent, TCurrent>(
            currentPipeline,
            _ => new IdentityPipeline<TCurrent>(),
            [decorator],
            suffixIsEmpty: true);
    }

    public IPipeline<TStart, TCurrent> Build(IServiceProvider serviceProvider)
    {
        IPipeline<TBoundary, TCurrent> suffix = _buildSuffix(serviceProvider);
        for (int index = _decorators.Count - 1; index >= 0; index--)
        {
            suffix = _decorators[index].Apply(serviceProvider, suffix);
        }

        return _buildPrefix is null
            ? (IPipeline<TStart, TCurrent>)(object)suffix
            : AppendedPipeline.Create(_buildPrefix(serviceProvider), suffix);
    }
}
