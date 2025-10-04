using FAI.Core.Abstractions;
using FAI.Core.ResultTypes;
using FAI.NLP.InferenceTasks.TextMultipleChoice;
using FAI.NLP.Tokenization;

namespace Example.MultipleChoice.Model;

public record struct SwagInput(string Context, string Text, string[] Endings);

public class SwagMultipleChoiceInference : IInference<SwagInput, ChoiceResult<TokenizedText>>
{
    private readonly IPipeline<TextMultipleChoiceInput, ChoiceResult<TokenizedText>> _pipeline;

    public SwagMultipleChoiceInference(IPipeline<TextMultipleChoiceInput, ChoiceResult<TokenizedText>> pipeline)
    {
        _pipeline = pipeline;
    }

    public async Task<ChoiceResult<TokenizedText>> Predict(SwagInput input)
    {
        var pipelineInput = MapSwagInputToPipelineInput(input);
        return await _pipeline.Predict(pipelineInput);
    }

    public async Task<ChoiceResult<TokenizedText>[]> BatchPredict(ReadOnlyMemory<SwagInput> input)
    {
        var output = new ChoiceResult<TokenizedText>[input.Length];
        await BatchPredict(input, output);
        return output;
    }

    public async Task BatchPredict(ReadOnlyMemory<SwagInput> input, Memory<ChoiceResult<TokenizedText>> output)
    {
        var pipelineInput = new TextMultipleChoiceInput[input.Length];
        ReadOnlySpan<SwagInput> inputSpan = input.Span;
        for (int i = 0; i < inputSpan.Length; i++)
        {
            pipelineInput[i] = MapSwagInputToPipelineInput(inputSpan[i]);
        }

        await _pipeline.BatchPredict(pipelineInput, output);
    }

    private static TextMultipleChoiceInput MapSwagInputToPipelineInput(SwagInput input)
    {
        var choices = new TokenizedText[input.Endings.Length];
        for (int i = 0; i < choices.Length; i++)
        {
            choices[i] = input.Text + " " + input.Endings[i];
        }

        var pipelineInput = new TextMultipleChoiceInput(input.Context, choices);
        return pipelineInput;
    }
}
