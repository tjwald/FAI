using System.Numerics.Tensors;

namespace FAI.Core.Pipelines;

public interface INamedTensorCollection : IEnumerable<KeyValuePair<string, Tensor<long>>>
{
    int Count { get; }

    int BatchSize { get; }

    int MaxTokenCount { get; }

    KeyValuePair<string, Tensor<long>> this[int index] { get; }

    bool TryGetValue(string name, out Tensor<long> tensor);

    string[] Names();

    Tensor<long>[] Tensors();

    Tensor<long>[] ToArray();
}
