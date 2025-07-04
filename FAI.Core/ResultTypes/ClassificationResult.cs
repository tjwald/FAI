namespace FAI.Core.ResultTypes;

/// <summary>
/// Represents the result of a classification operation.
/// </summary>
/// <typeparam name="T">The type of the classification choice.</typeparam>
/// <param name="Choice">The selected classification choice.</param>
/// <param name="Score">The confidence score of the classification.</param>
/// <param name="Logits">Optional raw logits from the classification model.</param>
public record struct ClassificationResult<T>(T Choice, float Score, float[]? Logits = null);