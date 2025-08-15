using System.Numerics.Tensors;

namespace FAI.Core.Abstractions;

public interface IPreprocessor<TInput, out TPreprocessContainer, TFloat> where TPreprocessContainer : IEnumerable<Tensor<TFloat>>
{
    TPreprocessContainer Preprocess(ReadOnlySpan<TInput> input);
}