using FAI.Core.Abstractions;

namespace FAI.Core.Extensions.DI;

public class PartitionBatchExecutorBuilder<TInput, TOutput>
{
    private Func<IServiceProvider, IBatchSlicer<TInput>>? _slicerFactory;
    private Func<IServiceProvider, IBatchSchedular<TInput, TOutput>>? _schedularFactory;
    private readonly IServiceCollection _serviceCollection;

    public PartitionBatchExecutorBuilder(IServiceCollection serviceCollection)
    {
        _serviceCollection = serviceCollection;
    }

    public PartitionBatchExecutorBuilder<TInput, TOutput> AddServices(Action<IServiceCollection> action)
    {
        action(_serviceCollection);
        return this;
    }

    public PartitionBatchExecutorBuilder<TInput, TOutput> WithSlicer(Func<IServiceProvider, IBatchSlicer<TInput>> slicerFactory)
    {
        _slicerFactory = slicerFactory;
        return this;
    }

    public PartitionBatchExecutorBuilder<TInput, TOutput> WithSchedular(Func<IServiceProvider, IBatchSchedular<TInput, TOutput>> schedularFactory)
    {
        _schedularFactory = schedularFactory;
        return this;
    }

    internal IBatchSlicer<TInput> BuildSlicer(IServiceProvider sp)
    {
        return _slicerFactory is not null ? _slicerFactory(sp) : sp.GetRequiredService<IBatchSlicer<TInput>>();
    }

    internal IBatchSchedular<TInput, TOutput> BuildSchedular(IServiceProvider sp)
    {
        return _schedularFactory is not null ? _schedularFactory(sp) : sp.GetRequiredService<IBatchSchedular<TInput, TOutput>>();
    }
}
