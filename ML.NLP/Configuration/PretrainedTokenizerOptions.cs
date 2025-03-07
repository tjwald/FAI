namespace ML.NLP.Configuration;

public enum TruncationOption
{
    Longest,
    Context,
    Text,
}

public record PretrainedTokenizerOptions(int PaddingToken, int MaxTokenLength = 512, TruncationOption TruncationOption = TruncationOption.Longest);