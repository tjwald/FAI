using FAI.Core.ResultTypes;
using FAI.NLP.InferenceTasks.TextClassification;
using FAI.NLP.Tokenization;

namespace FAI.NLP.Configuration;

/// <summary>
/// Represents configuration options for text classification tasks.
/// </summary>
/// <typeparam name="TClassification">The type of classification labels.</typeparam>
/// <param name="Choices">
/// An array of possible classification labels. The model output label number is indexed into this array.
/// </param>
/// <param name="StoreLogits">
/// Indicates whether to store raw model logits for further analysis. Defaults to <c>false</c>. This allows the Inference Task to reduce allocations if possible.  
/// </param>
public record TextClassificationOptions<TClassification>(TClassification[] Choices, bool StoreLogits = false);

public class TextClassificationBuilder<TClassification>
    : TextInferenceStepsBuilder<TokenizedText, ClassificationResult<TClassification>, TextClassification<TClassification>,
        TextClassificationBuilder<TClassification>>
{
    public TClassification[]? Choices { get; set; }
    public bool StoreLogits { get; set; } = false;

    public TextClassificationBuilder<TClassification> UseChoices(params TClassification[] choices)
    {
        Choices = choices;
        return this;
    }

    public override async ValueTask<TextClassification<TClassification>> BuildAsync()
    {
        var tokenizerTask = GetTokenizer();
        var modelExecutorTask = ExecutorFactory();

        ArgumentNullException.ThrowIfNull(Choices, nameof(Choices));

        return new TextClassification<TClassification>(await tokenizerTask, await modelExecutorTask, new(Choices, StoreLogits));
    }
}