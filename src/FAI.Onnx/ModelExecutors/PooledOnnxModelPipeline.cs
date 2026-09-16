using System.Numerics.Tensors;
using FAI.Core.ModelExecutors;
using FAI.Core.Pipelines;

namespace FAI.Onnx.ModelExecutors;

public sealed class PooledOnnxModelPipeline : IPipeline<BatchEncode, TensorOutputs<float>>
{
    private readonly IObjectPool<OnnxModelExecutorBase> _pool;

    public PooledOnnxModelPipeline(IObjectPool<OnnxModelExecutorBase> pool)
    {
        _pool = pool;
    }

    public ValueTask<TensorOutputs<float>> ExecuteAsync(
        BatchEncode input,
        CancellationToken cancellationToken = default)
        => _pool.Get().ExecuteAsync(input, cancellationToken);
}
