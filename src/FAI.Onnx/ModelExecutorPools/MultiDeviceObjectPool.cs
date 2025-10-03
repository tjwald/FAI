using FAI.Core.ModelExecutors;
using FAI.Onnx.ModelExecutors;
using FAI.Onnx.Utils;

namespace FAI.Onnx.ModelExecutorPools;

public class MultiDeviceObjectPool : IObjectPool<OnnxModelExecutorBase>
{
    private readonly List<OnnxModelExecutorBase> _pool;
    private readonly CircularAtomicCounter _counter;

    public MultiDeviceObjectPool(List<OnnxModelExecutorBase> pool)
    {
        _pool = pool;
        _counter = new CircularAtomicCounter(pool.Count);
    }

    public OnnxModelExecutorBase Get()
    {
        return _pool[_counter.Next()];
    }
}
