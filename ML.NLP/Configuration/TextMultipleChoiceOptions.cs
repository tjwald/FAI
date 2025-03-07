namespace ML.NLP.Configuration;

public sealed record TextMultipleChoiceOptions(int MaxChoices, bool StoreLogits = false);