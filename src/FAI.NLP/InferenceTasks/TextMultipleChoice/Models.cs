using FAI.NLP.Tokenization;

namespace FAI.NLP.InferenceTasks.TextMultipleChoice;

/// <summary>
/// Represents the input for a multiple-choice text classification task.
/// </summary>
/// <param name="Context">The context or prompt that applies to the multiple-choice options.</param>
/// <param name="Choices">The possible raw text choices.</param>
public sealed record TextMultipleChoiceInput(string Context, string[] Choices);

public sealed record TokenizedTextMultipleChoiceInput(string Context, TokenizedText[] Choices) : ITokenizable
{
    /// <summary>
    /// Gets the total number of tokens across all choices.
    /// Throws an exception if the input has not been tokenized.
    /// </summary>
    /// <exception cref="InvalidOperationException">Thrown if the input text has not been tokenized.</exception>
    public int TokenCount { get; } = Choices.Sum(choice => choice.TokenCount);

    /// <summary>
    /// Gets the maximum token length among all choices.
    /// Throws an exception if the input has not been tokenized.
    /// </summary>
    /// <exception cref="InvalidOperationException">Thrown if the input text has not been tokenized.</exception>
    public int MaxTokenLength { get; } = Choices.Max(choice => choice.TokenCount);

    /// <summary>
    /// Gets the number of choices available for classification.
    /// </summary>
    public int SentenceCount => Choices.Length;

}
