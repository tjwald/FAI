using System.Collections;
using System.Numerics.Tensors;

namespace FAI.Core.Pipelines;

/// <summary>
/// Hugging Face-style tokenization batch payload with named ONNX model inputs.
/// </summary>
/// <param name="InputIds">Token IDs tensor (<c>input_ids</c>).</param>
/// <param name="AttentionMask">Attention mask tensor (<c>attention_mask</c>).</param>
/// <param name="TokenTypeIds">Optional segment IDs tensor (<c>token_type_ids</c>).</param>
public readonly record struct BatchEncode(Tensor<long> InputIds, Tensor<long> AttentionMask, Tensor<long>? TokenTypeIds = null) : IEnumerable<KeyValuePair<string, Tensor<long>>>
{
    public const string InputIdsName = "input_ids";
    public const string AttentionMaskName = "attention_mask";
    public const string TokenTypeIdsName = "token_type_ids";

    public int BatchSize => (int)InputIds.Lengths[0];

    public int MaxTokenCount => (int)InputIds.Lengths[InputIds.Rank - 1];

    public int Count => TokenTypeIds is null ? 2 : 3;

    public Tensor<long>[] ToArray()
    {
        return TokenTypeIds is null
            ? [InputIds, AttentionMask]
            : [InputIds, AttentionMask, TokenTypeIds];
    }

    public IEnumerator<KeyValuePair<string, Tensor<long>>> GetEnumerator()
    {
        yield return new KeyValuePair<string, Tensor<long>>(InputIdsName, InputIds);
        yield return new KeyValuePair<string, Tensor<long>>(AttentionMaskName, AttentionMask);
        if (TokenTypeIds is not null)
        {
            yield return new KeyValuePair<string, Tensor<long>>(TokenTypeIdsName, TokenTypeIds);
        }
    }

    IEnumerator IEnumerable.GetEnumerator()
    {
        return GetEnumerator();
    }

    public static BatchEncode FromArray(Tensor<long>[] input)
    {
        return input.Length switch
        {
            2 => new BatchEncode(input[0], input[1]),
            3 => new BatchEncode(input[0], input[1], input[2]),
            _ => throw new ArgumentException("BatchEncode requires 2 or 3 tensors (input_ids, attention_mask, optional token_type_ids).", nameof(input)),
        };
    }
}
