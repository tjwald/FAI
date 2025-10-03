namespace FAI.NLP.Configuration;

public sealed record TextMultipleChoiceOptions(int MaxChoices, bool StoreLogits = false)
{
    public TextMultipleChoiceOptions() : this(0) { }
}
