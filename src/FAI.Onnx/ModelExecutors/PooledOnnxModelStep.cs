using System.Numerics.Tensors;
using FAI.Core.ModelExecutors;
using FAI.Core.Steps;

namespace FAI.Onnx.ModelExecutors;

public sealed class PooledOnnxModelStep : IStep<Tensor<long>[], TensorOutputs<float>>
{
    private readonly IObjectPool<OnnxModelExecutorBase> _pool;

    public PooledOnnxModelStep(IObjectPool<OnnxModelExecutorBase> pool)
    {
        _pool = pool;
    }

    public ValueTask<TensorOutputs<float>> ExecuteAsync(
        Tensor<long>[] input,
        CancellationToken cancellationToken = default)
        => _pool.Get().ExecuteAsync(input, cancellationToken);
}
