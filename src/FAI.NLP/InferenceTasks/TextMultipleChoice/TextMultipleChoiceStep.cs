using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using FAI.Core;
using FAI.Core.ResultTypes;
using FAI.Core.Steps;
using FAI.NLP.Configuration;
using FAI.NLP.Tokenization;

namespace FAI.NLP.InferenceTasks.TextMultipleChoice;

public sealed class TextMultipleChoiceStep :
    IPreallocatingStep<ReadOnlyMemory<TextMultipleChoiceInput>, Memory<ChoiceResult<TokenizedText>>>
{
    private readonly PretrainedTokenizer _tokenizer;
    private readonly IStep<Tensor<long>[], TensorOutputs<float>> _modelStep;
    private readonly TextMultipleChoiceOptions _options;

    public TextMultipleChoiceStep(
        PretrainedTokenizer tokenizer,
        IStep<Tensor<long>[], TensorOutputs<float>> modelStep,
        TextMultipleChoiceOptions options)
    {
        _tokenizer = tokenizer;
        _modelStep = modelStep;
        _options = options;
    }

    public bool TryAllocateOutput(
        ReadOnlyMemory<TextMultipleChoiceInput> input,
        out Memory<ChoiceResult<TokenizedText>> output)
    {
        output = new ChoiceResult<TokenizedText>[input.Length];
        return true;
    }

    public async ValueTask<Memory<ChoiceResult<TokenizedText>>> ExecuteAsync(
        ReadOnlyMemory<TextMultipleChoiceInput> input,
        CancellationToken cancellationToken = default)
    {
        _ = TryAllocateOutput(input, out Memory<ChoiceResult<TokenizedText>> output);
        await ExecuteAsync(input, output, cancellationToken);
        return output;
    }

    public async ValueTask ExecuteAsync(
        ReadOnlyMemory<TextMultipleChoiceInput> input,
        Memory<ChoiceResult<TokenizedText>> output,
        CancellationToken cancellationToken = default)
    {
        if (input.Length != output.Length)
        {
            throw new ArgumentException("Input and output batch sizes must match.", nameof(output));
        }

        if (input.IsEmpty)
        {
            return;
        }

        cancellationToken.ThrowIfCancellationRequested();
        BatchTokenizedResult tokenized = Preprocess(input.Span);
        Tensor<long>[] modelInput = tokenized.ToArray();
        using TensorOutputs<float> modelOutput = await _modelStep.ExecuteAsync(modelInput, cancellationToken);
        Decode(input, modelOutput.GetOutput(0), output);
    }

    private BatchTokenizedResult Preprocess(ReadOnlySpan<TextMultipleChoiceInput> input)
    {
        int maxChoiceCount = 0;
        int maxTokenCount = 0;
        for (int inputIndex = 0; inputIndex < input.Length; inputIndex++)
        {
            TextMultipleChoiceInput item = input[inputIndex];
            if (item.Choices.Length > _options.MaxChoices)
            {
                throw new InvalidOperationException($"Too many choices for text: {item.Context}");
            }

            maxChoiceCount = Math.Max(maxChoiceCount, item.Choices.Length);
            foreach (TokenizedText choice in item.Choices)
            {
                choice.Tokens ??= _tokenizer.Tokenize(item.Context, choice.Text);
                maxTokenCount = Math.Max(maxTokenCount, choice.TokenCount);
            }
        }

        Tensor<long> tokens = Tensor.CreateFromShape<long>([input.Length * maxChoiceCount, maxTokenCount]);
        Tensor<long> mask = Tensor.CreateFromShape<long>([input.Length * maxChoiceCount, maxTokenCount]);
        TensorDimensionSpan<long> tokenRows = tokens.GetDimensionSpan(0);
        TensorDimensionSpan<long> maskRows = mask.GetDimensionSpan(0);

        for (int inputIndex = 0; inputIndex < input.Length; inputIndex++)
        {
            for (int choiceIndex = 0; choiceIndex < input[inputIndex].Choices.Length; choiceIndex++)
            {
                List<int> choiceTokens = input[inputIndex].Choices[choiceIndex].Tokens!;
                int rowIndex = inputIndex * maxChoiceCount + choiceIndex;
                Span<long> tokenRow = tokenRows[rowIndex].AsSpan();
                Span<long> maskRow = maskRows[rowIndex].AsSpan();
                TensorPrimitives.ConvertChecked(CollectionsMarshal.AsSpan(choiceTokens), tokenRow);
                maskRow[..choiceTokens.Count].Fill(1);
            }
        }

        nint[] shape = [input.Length, maxChoiceCount, maxTokenCount];
        return new BatchTokenizedResult(tokens.Reshape(shape), mask.Reshape(shape));
    }

    private ChoiceResult<TokenizedText> GetMultipleChoiceResult(
        TextMultipleChoiceInput input,
        ReadOnlySpan<float> logits)
    {
        Span<float> probabilities = stackalloc float[logits.Length];
        TensorPrimitives.SoftMax(logits, probabilities);
        int choiceIndex = TensorPrimitives.IndexOfMax(probabilities);
        float[]? storedLogits = _options.StoreLogits ? logits.ToArray() : null;
        return new ChoiceResult<TokenizedText>(
            input.Choices[choiceIndex],
            choiceIndex,
            probabilities[choiceIndex],
            storedLogits);
    }

    private void Decode(
        ReadOnlyMemory<TextMultipleChoiceInput> input,
        ReadOnlyTensorSpan<float> tensor,
        Memory<ChoiceResult<TokenizedText>> output)
    {
        int rowCount = checked((int)tensor.Lengths[0]);
        if (rowCount != input.Length)
        {
            throw new InvalidOperationException(
                $"The model produced {rowCount} result rows for an input batch of {input.Length}.");
        }

        int rowIndex = 0;
        foreach (ReadOnlyTensorSpan<float> row in tensor.GetDimensionSpan(0))
        {
            int choiceCount = input.Span[rowIndex].Choices.Length;
            output.Span[rowIndex] = GetMultipleChoiceResult(
                input.Span[rowIndex],
                row.AsSpan()[..choiceCount]);
            rowIndex++;
        }
    }
}
