using System.Numerics.Tensors;
using ML.Infra;
using ML.Infra.Abstractions;
using ML.Infra.ResultTypes;
using ML.NLP.Configuration;
using ML.NLP.Tokenization;

namespace ML.NLP.InferenceTasks.TextMultipleChoice;


public class TextMultipleChoiceTask : InferenceSteps<TextMultipleChoiceInput, BatchTokenizedResult, ChoiceResult<TokenizedText>[], ChoiceResult<TokenizedText>>
{
    private readonly PretrainedTokenizer _tokenizer;
    private readonly TextMultipleChoiceOptions _options;
    private readonly IModelExecutor<long, float> _modelExecutor;

    public TextMultipleChoiceTask(PretrainedTokenizer tokenizer, IModelExecutor<long, float> modelExecutor, TextMultipleChoiceOptions options)
    {
        _tokenizer = tokenizer;
        _options = options;
        _modelExecutor = modelExecutor;
    }

    public override BatchTokenizedResult Preprocess(ReadOnlySpan<TextMultipleChoiceInput> input)
    {
        (List<List<int>?> tokensList, int maxChoiceCount, int maxTokenCount) =
            input[0].IsTokenized ? FlattenTokensWithPadding(input) : FlattenBatchTokenize(input);

        Tensor<long> tokenTensor = Tensor.Create<long>([input.Length * maxChoiceCount, maxTokenCount]);
        Tensor<long> maskTensor = Tensor.Create<long>([input.Length * maxChoiceCount, maxTokenCount]);

        TensorSpan<long> tokenTensorSpan = tokenTensor.AsTensorSpan();
        TensorSpan<long> maskTensorSpan = maskTensor.AsTensorSpan();

        int outputRow = 0;
        foreach (List<int>? tokens in tokensList)
        {
            if (tokens is null)
            {
                // skip writing tokens and mask since mask = 0 by default, and tokens don't matter when mask = 0
                outputRow += _options.MaxChoices - outputRow % _options.MaxChoices;
                continue;
            }

            Span<long> tokenRow = tokenTensorSpan.GetRowSpan(outputRow);
            Span<long> maskRow = maskTensorSpan.GetRowSpan(outputRow);
            for (int tokenIndex = 0; tokenIndex < tokens.Count; tokenIndex++)
            {
                tokenRow[tokenIndex] = tokens[tokenIndex];
            }

            maskRow[..tokens.Count].Fill(1);
            outputRow++;
        }

        Span<nint> shape = [input.Length, maxChoiceCount, -1];
        return new BatchTokenizedResult(tokenTensor.Reshape(shape), maskTensor.Reshape(shape));
    }

    private (List<List<int>?> tokensList, int maxChoiceCount, int maxTokenCount) FlattenBatchTokenize(ReadOnlySpan<TextMultipleChoiceInput> input)
    {
        List<List<int>?> tokens = [];
        int maxTokenCount = 0;
        int maxChoiceCount = 0;
        foreach ((string context, TokenizedText[] choices) in input)
        {
            if (choices.Length > _options.MaxChoices)
            {
                throw new InvalidOperationException($"Too many choices for text: {context}");
            }

            maxChoiceCount = Math.Max(maxChoiceCount, choices.Length);
            foreach (var choice in choices)
            {
                List<int> tokenizedText = _tokenizer.Tokenize(context, choice.Text);
                choice.Tokens = tokenizedText;
                maxTokenCount = Math.Max(maxTokenCount, tokenizedText.Count);
                tokens.Add(tokenizedText);
            }

            if (_options.MaxChoices > choices.Length)
            {
                tokens.Add(null);
            }
        }

        return (tokens, maxChoiceCount, maxTokenCount);
    }

    private (List<List<int>?> tokensList, int maxChoiceCount, int maxTokenCount) FlattenTokensWithPadding(ReadOnlySpan<TextMultipleChoiceInput> input)
    {
        List<List<int>?> tokens = [];
        int maxTokenCount = 0;
        int maxChoiceCount = 0;
        foreach (var t in input)
        {
            TokenizedText[] choices = t.Choices;
            maxChoiceCount = Math.Max(maxChoiceCount, choices.Length);
            foreach (var choice in choices)
            {
                maxTokenCount = Math.Max(maxTokenCount, choice.TokenCount);
                tokens.Add(choice.Tokens!);
            }

            if (choices.Length % _options.MaxChoices > 0)
            {
                tokens.Add(null);
            }
        }

        return (tokens, maxChoiceCount, maxTokenCount);
    }

    public override async Task<ChoiceResult<TokenizedText>[]> RunModel(ReadOnlyMemory<TextMultipleChoiceInput> input, BatchTokenizedResult tokenizedResult)
    {
        var outputs = new ChoiceResult<TokenizedText>[input.Length];

        await _modelExecutor.RunAsync([tokenizedResult.Tokens, tokenizedResult.Mask], (logits, _) =>
        {
            for (int indexInBatch = 0; indexInBatch < logits.Lengths[0]; indexInBatch++)
            {
                ReadOnlySpan<float> rowLogits = logits.GetRowSpan(indexInBatch);
                outputs[indexInBatch] = GetMultipleChoiceResult(input.Span[indexInBatch], rowLogits);
            }
        });
        return outputs;
    }

    public override void PostProcess(ReadOnlySpan<TextMultipleChoiceInput> inputs, BatchTokenizedResult preprocesses, ChoiceResult<TokenizedText>[] modelOutput,
        Span<ChoiceResult<TokenizedText>> outputs)
    {
        modelOutput.AsSpan().CopyTo(outputs);
    }

    private ChoiceResult<TokenizedText> GetMultipleChoiceResult(TextMultipleChoiceInput input, ReadOnlySpan<float> logits)
    {
        Span<float> probabilities = stackalloc float[logits.Length];
        TensorPrimitives.SoftMax(logits, probabilities);
        int argmax = TensorPrimitives.IndexOfMax<float>(probabilities);
        float score = TensorPrimitives.Max<float>(probabilities);

        float[]? logitsArray = _options.StoreLogits ? logits.ToArray() : null;

        return new ChoiceResult<TokenizedText>(input.Choices[argmax], argmax, score, logitsArray);
    }
}