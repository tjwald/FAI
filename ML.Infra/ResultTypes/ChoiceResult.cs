namespace ML.Infra.ResultTypes;

public sealed record ChoiceResult<TChoice>(TChoice Choice, int ChoiceIndex, float Score, float[]? Logits);