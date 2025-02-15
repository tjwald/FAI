namespace ML.NLP.Configuration;

public record TextClassificationOptions<TClassification>(TClassification[] Choices, bool StoreLogits = false);