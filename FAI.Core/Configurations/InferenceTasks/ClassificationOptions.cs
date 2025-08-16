namespace FAI.Core.Configurations.InferenceTasks;

public record ClassificationOptions<TClassification>(TClassification[] Choices, bool StoreLogits = false);
