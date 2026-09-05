using FAI.Core.Pipelines;
using Microsoft.Extensions.DependencyInjection.Extensions;

namespace FAI.Core.Extensions.DI;

public static class IndexedBatchServiceExtensions
{
    public static IServiceCollection AddIndexedBatchRegistry(this IServiceCollection services)
    {
        services.TryAddSingleton<IIndexedBatchRegistry, IndexedBatchRegistry>();
        return services;
    }

    public static IServiceCollection AddBatchOperations<TBatch, TOperations>(this IServiceCollection services)
        where TOperations : class, IWritableIndexedBatch<TBatch>
    {
        services.AddIndexedBatchRegistry();
        services.TryAddSingleton<IWritableIndexedBatch<TBatch>, TOperations>();
        services.TryAddSingleton<IReadOnlyIndexedBatch<TBatch>>(sp => sp.GetRequiredService<IWritableIndexedBatch<TBatch>>());
        return services;
    }

    public static IServiceCollection AddBatchOperations<TBatch>(this IServiceCollection services, IWritableIndexedBatch<TBatch> operations)
    {
        services.AddIndexedBatchRegistry();
        services.TryAddSingleton(operations);
        services.TryAddSingleton<IReadOnlyIndexedBatch<TBatch>>(operations);
        return services;
    }

    public static IServiceCollection AddMemoryBatch<T>(this IServiceCollection services)
        => services.AddBatchOperations<Memory<T>, MemoryBatchOperations<T>>();

    public static IServiceCollection AddReadOnlyMemoryBatch<T>(this IServiceCollection services)
    {
        services.AddIndexedBatchRegistry();
        services.TryAddSingleton<IReadOnlyIndexedBatch<ReadOnlyMemory<T>>, ReadOnlyMemoryBatchOperations<T>>();
        return services;
    }

    public static IServiceCollection AddTensorBatch<T>(this IServiceCollection services)
    {
        services.AddIndexedBatchRegistry();
        services.TryAddSingleton<TensorBatchOperations<T>>();
        services.TryAddSingleton<IWritableIndexedBatch<System.Numerics.Tensors.Tensor<T>>>(sp => sp.GetRequiredService<TensorBatchOperations<T>>());
        services.TryAddSingleton<IReadOnlyIndexedBatch<System.Numerics.Tensors.Tensor<T>>>(sp => sp.GetRequiredService<TensorBatchOperations<T>>());
        return services;
    }

    public static IServiceCollection AddDefaultIndexedBatches(this IServiceCollection services)
    {
        services.AddIndexedBatchRegistry();
        services.AddMemoryBatch<int>();
        services.AddMemoryBatch<long>();
        services.AddMemoryBatch<float>();
        services.AddMemoryBatch<double>();
        services.AddMemoryBatch<string>();
        services.AddMemoryBatch<bool>();
        services.AddReadOnlyMemoryBatch<int>();
        services.AddReadOnlyMemoryBatch<long>();
        services.AddReadOnlyMemoryBatch<float>();
        services.AddReadOnlyMemoryBatch<string>();
        services.AddTensorBatch<float>();
        services.AddTensorBatch<long>();
        return services;
    }
}
