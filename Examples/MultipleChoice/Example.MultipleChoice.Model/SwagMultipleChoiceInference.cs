using FAI.Core.Abstractions;
using FAI.Core.ResultTypes;
using FAI.Core.Steps;
using FAI.NLP.InferenceTasks.TextMultipleChoice;
using FAI.NLP.Tokenization;

namespace Example.MultipleChoice.Model;

public record struct SwagInput(string Context, string Text, string[] Endings);

public class SwagMultipleChoiceInference : IInference<SwagInput, ChoiceResult<TokenizedText>>
{
    private readonly IStep<ReadOnlyMemory<TextMultipleChoiceInput>, Memory<ChoiceResult<TokenizedText>>> _pipeline;

    public SwagMultipleChoiceInference(
        IStep<ReadOnlyMemory<TextMultipleChoiceInput>, Memory<ChoiceResult<TokenizedText>>> pipeline)
    {
        _pipeline = pipeline;
    }

    public async Task<ChoiceResult<TokenizedText>> Predict(SwagInput input)
    {
        ChoiceResult<TokenizedText>[] output = await BatchPredict(new[] { input });
        return output[0];
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

        Memory<ChoiceResult<TokenizedText>> results = await _pipeline.ExecuteAsync(pipelineInput);
        results.CopyTo(output);
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
