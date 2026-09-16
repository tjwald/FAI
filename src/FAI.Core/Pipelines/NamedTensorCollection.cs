using System.Collections;
using System.Numerics.Tensors;

namespace FAI.Core.Pipelines;

public readonly record struct NamedTensorCollection : INamedTensorCollection
{
    private readonly KeyValuePair<string, Tensor<long>>[] _inputs;

    public NamedTensorCollection(params Tensor<long>[] inputs)
    {
        if (inputs.Length == 0)
        {
            throw new ArgumentException("At least one tensor input is required.", nameof(inputs));
        }

        _inputs = new KeyValuePair<string, Tensor<long>>[inputs.Length];
        for (int i = 0; i < inputs.Length; i++)
        {
            _inputs[i] = new KeyValuePair<string, Tensor<long>>($"input_{i}", inputs[i]);
        }
    }

    public NamedTensorCollection(params KeyValuePair<string, Tensor<long>>[] inputs)
    {
        if (inputs.Length == 0)
        {
            throw new ArgumentException("At least one named tensor input is required.", nameof(inputs));
        }

        _inputs = inputs;
    }

    public int Count => _inputs?.Length ?? 0;

    public int BatchSize => (int)this[0].Value.Lengths[0];

    public int MaxTokenCount => (int)this[0].Value.Lengths[this[0].Value.Rank - 1];

    public KeyValuePair<string, Tensor<long>> this[int index] => (_inputs ?? throw new InvalidOperationException("No named tensor inputs available."))[index];

    public bool TryGetValue(string name, out Tensor<long> tensor)
    {
        if (_inputs is null)
        {
            tensor = default!;
            return false;
        }

        for (int i = 0; i < _inputs.Length; i++)
        {
            if (_inputs[i].Key == name)
            {
                tensor = _inputs[i].Value;
                return true;
            }
        }

        tensor = default!;
        return false;
    }

    public string[] Names()
    {
        var names = new string[Count];
        for (int i = 0; i < names.Length; i++)
        {
            names[i] = _inputs[i].Key;
        }

        return names;
    }

    public Tensor<long>[] Tensors()
    {
        var tensors = new Tensor<long>[Count];
        for (int i = 0; i < tensors.Length; i++)
        {
            tensors[i] = _inputs[i].Value;
        }

        return tensors;
    }

    public Tensor<long>[] ToArray()
    {
        return Tensors();
    }

    public IEnumerator<KeyValuePair<string, Tensor<long>>> GetEnumerator()
    {
        if (_inputs is null)
        {
            return Enumerable.Empty<KeyValuePair<string, Tensor<long>>>().GetEnumerator();
        }

        return ((IEnumerable<KeyValuePair<string, Tensor<long>>>)_inputs).GetEnumerator();
    }

    IEnumerator IEnumerable.GetEnumerator()
    {
        return GetEnumerator();
    }
}
