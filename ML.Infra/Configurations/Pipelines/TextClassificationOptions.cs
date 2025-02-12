namespace ML.Infra.Configurations.Pipelines;

public record TextClassificationOptions<TClassification>(TClassification[] Choices, bool StoreLogits = false);