using System.Numerics;
using System.Numerics.Tensors;
using FAI.Core.Configurations.InferenceTasks;

namespace FAI.Core.ResultTypes;

public static class ClassificationTensorExtensions
{
    public static ClassificationResult<TClassification, TScore> GetClassificationResult<TClassification, TScore>(
        this ClassificationOptions<TClassification> options,
        ReadOnlySpan<TScore> logits)
        where TScore : unmanaged, IFloatingPointIeee754<TScore>
    {
        Span<TScore> probabilities = stackalloc TScore[logits.Length];
        TensorPrimitives.SoftMax(logits, probabilities);
        int choiceIndex = TensorPrimitives.IndexOfMax(probabilities);
        TScore[]? storedLogits = options.StoreLogits ? logits.ToArray() : null;

        return new ClassificationResult<TClassification, TScore>(
            options.Choices[choiceIndex],
            probabilities[choiceIndex],
            storedLogits);
    }
}
