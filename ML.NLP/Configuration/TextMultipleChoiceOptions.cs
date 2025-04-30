using ML.Infra.ResultTypes;
using ML.NLP.InferenceTasks.TextMultipleChoice;
using ML.NLP.Tokenization;

namespace ML.NLP.Configuration;

public sealed record TextMultipleChoiceOptions(int MaxChoices, bool StoreLogits = false);

public class TextMultipleChoiceBuilder
    : TextInferenceStepsBuilder<TextMultipleChoiceInput, ChoiceResult<TokenizedText>, TextMultipleChoiceTask, TextMultipleChoiceBuilder>
{
    public int MaxChoices { get; set; }
    public bool StoreLogits { get; set; } = false;

    public override async ValueTask<TextMultipleChoiceTask> BuildAsync()
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(MaxChoices);

        var tokenizerTask = GetTokenizer();
        var modelExecutorTask = ExecutorFactory();

        return new TextMultipleChoiceTask(await tokenizerTask, await modelExecutorTask, new(MaxChoices, StoreLogits));
    }
}