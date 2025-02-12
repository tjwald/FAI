namespace ML.Infra.ResultTypes;

public record struct ClassificationResult<T>(T Choice, float Score, float[]? Logits = null);
