using ML.Infra.ModelExecutors;
using ML.Onnx.ModelExecutors;
using ML.Onnx.Utils;

namespace ML.Onnx.ModelExecutorPools;

public class MultiDeviceObjectPool: IObjectPool<OnnxModelExecutorBase>
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