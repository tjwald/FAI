namespace ML.NLP.Configuration;

/// <summary>
/// Represents configuration options for multiple-choice text classification tasks.
/// </summary>
/// <param name="MaxChoices">
/// The maximum number of choices allowed for a single classification request.
/// </param>
/// <param name="StoreLogits">
/// Indicates whether to store raw model logits for further analysis. Defaults to <c>false</c>. Can allow the inference task to reduce allocations.
/// </param>
public sealed record TextMultipleChoiceOptions(int MaxChoices, bool StoreLogits = false);
