namespace FAI.NLP.Configuration;

public sealed record MaxPaddedTokensPartitionerOptions(double MaxPaddedTokenRatio, int MaxTokenCount)
{
    public MaxPaddedTokensPartitionerOptions() : this(0, 0) { }
}
