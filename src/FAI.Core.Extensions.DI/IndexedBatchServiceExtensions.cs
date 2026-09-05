using FAI.Core.Pipelines;
using Microsoft.Extensions.DependencyInjection.Extensions;

namespace FAI.Core.Extensions.DI;

public static class IndexedBatchServiceExtensions
{
    public static IServiceCollection AddBatchOperations<TBatch, TOperations>(this IServiceCollection services)
        where TOperations : class, IWritableIndexedBatch<TBatch>
    {
        services.TryAddSingleton<IWritableIndexedBatch<TBatch>, TOperations>();
        services.TryAddSingleton<IReadOnlyIndexedBatch<TBatch>>(sp => sp.GetRequiredService<IWritableIndexedBatch<TBatch>>());
        return services;
    }

    public static IServiceCollection AddBatchOperations<TBatch>(this IServiceCollection services, IWritableIndexedBatch<TBatch> operations)
    {
        services.TryAddSingleton(operations);
        services.TryAddSingleton<IReadOnlyIndexedBatch<TBatch>>(operations);
        return services;
    }

    public static IServiceCollection AddMemoryBatch<T>(this IServiceCollection services)
        => services.AddBatchOperations<Memory<T>, MemoryBatchOperations<T>>();

    public static IServiceCollection AddReadOnlyMemoryBatch<T>(this IServiceCollection services)
    {
        services.TryAddSingleton<IReadOnlyIndexedBatch<ReadOnlyMemory<T>>, ReadOnlyMemoryBatchOperations<T>>();
        return services;
    }

    public static IServiceCollection AddTensorBatch<T>(this IServiceCollection services)
    {
        services.TryAddSingleton<TensorBatchOperations<T>>();
        services.TryAddSingleton<IWritableIndexedBatch<System.Numerics.Tensors.Tensor<T>>>(sp => sp.GetRequiredService<TensorBatchOperations<T>>());
        services.TryAddSingleton<IReadOnlyIndexedBatch<System.Numerics.Tensors.Tensor<T>>>(sp => sp.GetRequiredService<TensorBatchOperations<T>>());
        return services;
    }

    public static IWritableIndexedBatch<TBatch> GetRequiredWritableBatch<TBatch>(this IServiceProvider serviceProvider)
        => serviceProvider.GetService<IWritableIndexedBatch<TBatch>>()
            ?? throw new InvalidOperationException(
                $"No writable indexed batch operations are registered in the service collection for '{typeof(TBatch)}'. " +
                $"Register them using services.AddBatchOperations<{typeof(TBatch)}, ...>() or services.AddMemoryBatch<T>().");

    public static IReadOnlyIndexedBatch<TBatch> GetRequiredReadOnlyBatch<TBatch>(this IServiceProvider serviceProvider)
        => serviceProvider.GetService<IReadOnlyIndexedBatch<TBatch>>()
            ?? throw new InvalidOperationException(
                $"No read-only indexed batch operations are registered in the service collection for '{typeof(TBatch)}'. " +
                $"Register them using services.AddBatchOperations<{typeof(TBatch)}, ...>() or services.AddReadOnlyMemoryBatch<T>().");
}
