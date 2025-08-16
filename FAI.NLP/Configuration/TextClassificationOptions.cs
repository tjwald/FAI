using FAI.Core.ResultTypes;
using FAI.NLP.InferenceTasks.TextClassification;
using FAI.NLP.Tokenization;

namespace FAI.NLP.Configuration;

public class TextClassificationBuilder<TClassification>
    : TextInferenceStepsBuilder<TokenizedText, ClassificationResult<TClassification, float>, TextClassification<TClassification>,
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
