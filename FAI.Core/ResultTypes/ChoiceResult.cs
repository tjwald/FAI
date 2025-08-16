namespace FAI.Core.ResultTypes;

/// <summary>
/// Represents the result of a choice operation, containing the selected choice, its index, score, and optional logits.
/// </summary>
/// <typeparam name="TChoice">The type of the choice.</typeparam>
/// <param name="Choice">The selected choice.</param>
/// <param name="ChoiceIndex">The index of the selected choice.</param>
/// <param name="Score">The score associated with the selected choice.</param>
/// <param name="Logits">Optional logits associated with the choice, if available.</param>
public sealed record ChoiceResult<TChoice>(TChoice Choice, int ChoiceIndex, float Score, float[]? Logits);
