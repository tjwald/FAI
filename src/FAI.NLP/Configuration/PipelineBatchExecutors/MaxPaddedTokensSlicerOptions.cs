namespace FAI.NLP.Configuration.PipelineBatchExecutors;

public sealed record MaxPaddedTokensSlicerOptions(double MaxPaddedTokenRatio, int MaxTokenCount)
{
    public MaxPaddedTokensSlicerOptions() : this(0, 0) { }
}
