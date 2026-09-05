using System.Numerics;
using System.Numerics.Tensors;

namespace FAI.Core.Pipelines;

public abstract class TensorOutputs<T> : IDisposable
    where T : unmanaged, INumber<T>
{
    public abstract int Count { get; }

    public abstract ReadOnlyTensorSpan<T> GetOutput(int index);

    public abstract void Dispose();
}
