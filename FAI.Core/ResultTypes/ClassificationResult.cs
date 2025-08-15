using System.Numerics;

namespace FAI.Core.ResultTypes;

/// <summary>
/// Represents the result of a classification operation.
/// </summary>
/// <typeparam name="T">The type of the classification choice.</typeparam>
/// <typeparam name="TScore">The type of quantization used</typeparam>
/// <param name="Choice">The selected classification choice.</param>
/// <param name="Score">The confidence score of the classification.</param>
/// <param name="Logits">Optional raw logits from the classification model.</param>
public record struct ClassificationResult<T, TScore>(T Choice, TScore Score, TScore[]? Logits = null) where TScore : IFloatingPointIeee754<TScore>;