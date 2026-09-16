using System.Collections;
using System.Numerics.Tensors;

namespace FAI.Core.Pipelines;

public readonly record struct BatchEncode : INamedTensorCollection
{
    public const string InputIdsName = "input_ids";
    public const string AttentionMaskName = "attention_mask";
    public const string TokenTypeIdsName = "token_type_ids";

    private readonly NamedTensorCollection _inner;

    public BatchEncode(Tensor<long> inputIds, Tensor<long>? attentionMask = null, Tensor<long>? tokenTypeIds = null)
    {
        if (attentionMask is not null && tokenTypeIds is not null)
        {
            _inner = new NamedTensorCollection(
                new KeyValuePair<string, Tensor<long>>(InputIdsName, inputIds),
                new KeyValuePair<string, Tensor<long>>(AttentionMaskName, attentionMask),
                new KeyValuePair<string, Tensor<long>>(TokenTypeIdsName, tokenTypeIds));
            return;
        }

        if (attentionMask is not null)
        {
            _inner = new NamedTensorCollection(
                new KeyValuePair<string, Tensor<long>>(InputIdsName, inputIds),
                new KeyValuePair<string, Tensor<long>>(AttentionMaskName, attentionMask));
            return;
        }

        if (tokenTypeIds is not null)
        {
            _inner = new NamedTensorCollection(
                new KeyValuePair<string, Tensor<long>>(InputIdsName, inputIds),
                new KeyValuePair<string, Tensor<long>>(TokenTypeIdsName, tokenTypeIds));
            return;
        }

        _inner = new NamedTensorCollection(
            new KeyValuePair<string, Tensor<long>>(InputIdsName, inputIds));
    }

    public BatchEncode(NamedTensorCollection inner)
    {
        _inner = inner;
    }

    public int Count => _inner.Count;

    public int BatchSize => _inner.BatchSize;

    public int MaxTokenCount => _inner.MaxTokenCount;

    public KeyValuePair<string, Tensor<long>> this[int index] => _inner[index];

    public Tensor<long> InputIds => GetRequired(InputIdsName);

    public Tensor<long>? AttentionMask => TryGetValue(AttentionMaskName, out Tensor<long> tensor) ? tensor : null;

    public Tensor<long>? TokenTypeIds => TryGetValue(TokenTypeIdsName, out Tensor<long> tensor) ? tensor : null;

    public bool TryGetValue(string name, out Tensor<long> tensor) => _inner.TryGetValue(name, out tensor);

    public string[] Names() => _inner.Names();

    public Tensor<long>[] Tensors() => _inner.Tensors();

    public Tensor<long>[] ToArray() => _inner.ToArray();

    public IEnumerator<KeyValuePair<string, Tensor<long>>> GetEnumerator() => _inner.GetEnumerator();

    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

    public static implicit operator NamedTensorCollection(BatchEncode input) => input._inner;

    public static explicit operator BatchEncode(NamedTensorCollection input) => new(input);

    private Tensor<long> GetRequired(string name)
    {
        if (!_inner.TryGetValue(name, out Tensor<long> tensor))
        {
            throw new InvalidOperationException($"Missing required batch input '{name}'.");
        }

        return tensor;
    }
}
