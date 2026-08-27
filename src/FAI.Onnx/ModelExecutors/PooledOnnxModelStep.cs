using System.Numerics.Tensors;
using FAI.Core.ModelExecutors;
using FAI.Core.Steps;

namespace FAI.Onnx.ModelExecutors;

public sealed class PooledOnnxModelStep :
    IAllocatingStep<Tensor<long>[], Tensor<float>[]>,
    IBorrowedTensorProducer<Tensor<long>[], float>
{
    private readonly IObjectPool<OnnxModelExecutorBase> _pool;
    private readonly OnnxModelExecutorBase _metadataExecutor;

    public PooledOnnxModelStep(IObjectPool<OnnxModelExecutorBase> pool)
    {
        _pool = pool;
        _metadataExecutor = pool.Get();
    }

    public ValueTask<BatchLease<Tensor<float>[]>> RentOutputAsync(
        Tensor<long>[] input,
        CancellationToken cancellationToken = default)
        => _metadataExecutor.RentOutputAsync(input, cancellationToken);

    public ValueTask ExecuteAsync(
        Tensor<long>[] input,
        Tensor<float>[] output,
        CancellationToken cancellationToken = default)
        => _pool.Get().ExecuteAsync(input, output, cancellationToken);

    public ValueTask ExecuteAsync<TOutput>(
        Tensor<long>[] input,
        TOutput output,
        IBorrowedTensorConsumer<float, TOutput> consumer,
        CancellationToken cancellationToken = default)
        => _pool.Get().ExecuteAsync(input, output, consumer, cancellationToken);
}
