namespace FAI.NLP.Configuration;

/// <summary>
/// Defines the truncation strategy to use when Context and Text is longer than allowed input length to model.
/// Truncation is removing tokens from the end of the sequence.
/// </summary>
public enum TruncationOption
{
    /// <summary>
    /// Will truncate the longer part of the context or text.
    /// </summary>
    Longest,

    /// <summary>
    /// Will always truncate the context.
    /// </summary>
    Context,

    /// <summary>
    /// Will always truncate the text.
    /// </summary>
    Text,
}

/// <summary>
/// Represents the configuration options for a pretrained tokenizer.
/// </summary>
/// <param name="PaddingToken">The token used for padding sequences.</param>
/// <param name="MaxTokenLength">The maximum number of tokens allowed in a sequence.</param>
/// <param name="TruncationOption">The truncation strategy applied when tokenizing text.</param>
public record PretrainedTokenizerOptions(int PaddingToken = 0, int MaxTokenLength = 512, TruncationOption TruncationOption = TruncationOption.Longest);