using System.Numerics;
using System.Numerics.Tensors;
using FAI.Core.Configurations.InferenceTasks;
using FAI.Core.ResultTypes;

namespace FAI.Core.InferenceTasks.Classification;

public static class ClassificationTensorUtils
{
    /// <summary>
    /// Generates a classification result from the raw logits produced by the model.
    /// </summary>
    /// <param name="classificationOptions">The configuration containing the choices and the options for returning the result</param>
    /// <param name="logits">The raw logits produced by the model.</param>
    /// <returns>A classification result containing the predicted label and confidence score.</returns>
    public static ClassificationResult<TClassification, TScore> GetClassificationResult<TClassification, TScore>(
        this ClassificationOptions<TClassification> classificationOptions, ReadOnlySpan<TScore> logits) where TScore : unmanaged, IFloatingPointIeee754<TScore>
    {
        Span<TScore> probabilities = stackalloc TScore[logits.Length];
        TensorPrimitives.SoftMax(logits, probabilities);
        int argmax = TensorPrimitives.IndexOfMax<TScore>(probabilities);
        var score = probabilities[argmax];

        TScore[]? logitsArray = classificationOptions.StoreLogits ? logits.ToArray() : null;

        return new ClassificationResult<TClassification, TScore>(classificationOptions.Choices[argmax], score, logitsArray);
    }
}