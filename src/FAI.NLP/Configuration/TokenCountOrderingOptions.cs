namespace FAI.NLP.Configuration;

public sealed record TokenCountOrderingOptions(bool Ascending)
{
    public TokenCountOrderingOptions() : this(true) { }
}
