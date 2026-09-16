using System.Numerics.Tensors;
using FAI.Core.Pipelines;
using FAI.NLP.Tokenization;

namespace FAI.NLP.Tests.Tokenization;

public sealed class BatchEncodeTests
{
    [Fact]
    public void BatchEncode_ExposesHuggingFaceNamedInputs()
    {
        Tensor<long> inputIds = Tensor.Create([1L, 2L, 3L, 4L], [2, 2]);
        Tensor<long> attentionMask = Tensor.Create([1L, 1L, 1L, 0L], [2, 2]);
        Tensor<long> tokenTypeIds = Tensor.Create([0L, 0L, 0L, 0L], [2, 2]);
        var batchEncode = new BatchEncode(inputIds, attentionMask, tokenTypeIds);

        Assert.Equal(3, batchEncode.Count);
        Assert.Same(inputIds, batchEncode.InputIds);
        Assert.Same(attentionMask, batchEncode.AttentionMask);
        Assert.Same(tokenTypeIds, batchEncode.TokenTypeIds);
    }

    [Fact]
    public void BatchEncode_ImplicitlyConvertsToNamedTensorCollection()
    {
        Tensor<long> inputIds = Tensor.Create([1L, 2L, 3L], [1, 3]);
        var batchEncode = new BatchEncode(inputIds);

        NamedTensorCollection namedInputs = batchEncode;

        Assert.Equal(1, namedInputs.Count);
        Assert.True(namedInputs.TryGetValue(BatchEncode.InputIdsName, out Tensor<long> resolved));
        Assert.Same(inputIds, resolved);
    }

    [Fact]
    public void BatchEncode_FromNamedTensorCollectionWithoutInputIds_ThrowsOnInputIdsAccess()
    {
        Tensor<long> tensor = Tensor.Create([1L, 2L], [1, 2]);
        var named = new NamedTensorCollection(new KeyValuePair<string, Tensor<long>>("input_0", tensor));
        BatchEncode batchEncode = (BatchEncode)named;

        Assert.Throws<InvalidOperationException>(() => batchEncode.InputIds);
    }
}
