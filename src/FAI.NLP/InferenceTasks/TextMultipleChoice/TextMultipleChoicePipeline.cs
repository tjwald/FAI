using System.Numerics.Tensors;
using FAI.Core;
using FAI.Core.Pipelines;
using FAI.Core.ResultTypes;
using FAI.NLP.Configuration;
using FAI.NLP.Tokenization;

namespace FAI.NLP.InferenceTasks.TextMultipleChoice;

public sealed class TextMultipleChoiceTensorization : IPipeline<ReadOnlyMemory<TokenizedTextMultipleChoiceInput>, Tensor<long>[]>
{
    private readonly TextMultipleChoiceOptions _options;

    public TextMultipleChoiceTensorization(TextMultipleChoiceOptions options)
    {
        _options = options;
    }

    public ValueTask<Tensor<long>[]> ExecuteAsync(
        ReadOnlyMemory<TokenizedTextMultipleChoiceInput> input,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (input.IsEmpty)
        {
            throw new ArgumentException("Cannot encode an empty multiple choice batch.", nameof(input));
        }

        return ValueTask.FromResult(Encode(input.Span));
    }

    private Tensor<long>[] Encode(ReadOnlySpan<TokenizedTextMultipleChoiceInput> input)
    {
        int maxChoiceCount = 0;
        int maxTokenCount = 0;
        for (int inputIndex = 0; inputIndex < input.Length; inputIndex++)
        {
            TokenizedTextMultipleChoiceInput item = input[inputIndex];
            if (item.Choices.Length > _options.MaxChoices)
            {
                throw new InvalidOperationException($"Too many choices for text: {item.Context}");
            }

            maxChoiceCount = Math.Max(maxChoiceCount, item.Choices.Length);
            foreach (TokenizedText choice in item.Choices)
            {
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
                ReadOnlySpan<int> choiceTokens = input[inputIndex].Choices[choiceIndex].Tokens.Span;
                int rowIndex = inputIndex * maxChoiceCount + choiceIndex;
                Span<long> tokenRow = tokenRows[rowIndex].AsSpan();
                Span<long> maskRow = maskRows[rowIndex].AsSpan();
                TensorPrimitives.ConvertChecked(choiceTokens, tokenRow);
                maskRow[..choiceTokens.Length].Fill(1);
            }
        }

        nint[] shape = [input.Length, maxChoiceCount, maxTokenCount];
        return new BatchTokenizedResult(tokens.Reshape(shape), mask.Reshape(shape)).ToArray();
    }
}

public sealed class TextMultipleChoiceDecoding :
    IDestinationPipeline<(ReadOnlyMemory<TokenizedTextMultipleChoiceInput> Input, TensorOutputs<float> ModelOutputs), Memory<ChoiceResult<TokenizedText>>>
{
    private readonly TextMultipleChoiceOptions _options;

    public TextMultipleChoiceDecoding(TextMultipleChoiceOptions options)
    {
        _options = options;
    }

    public async ValueTask<Memory<ChoiceResult<TokenizedText>>> ExecuteAsync(
        (ReadOnlyMemory<TokenizedTextMultipleChoiceInput> Input, TensorOutputs<float> ModelOutputs) input,
        CancellationToken cancellationToken = default)
    {
        if (input.ModelOutputs.Count == 0)
        {
            throw new InvalidOperationException("Multiple choice requires at least one model output.");
        }

        int rowCount = checked((int)input.ModelOutputs.GetOutput(0).Lengths[0]);
        Memory<ChoiceResult<TokenizedText>> output = new ChoiceResult<TokenizedText>[rowCount];
        await ExecuteAsync(input, output, cancellationToken);
        return output;
    }

    public ValueTask ExecuteAsync(
        (ReadOnlyMemory<TokenizedTextMultipleChoiceInput> Input, TensorOutputs<float> ModelOutputs) input,
        Memory<ChoiceResult<TokenizedText>> destination,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (input.Input.Length != destination.Length)
        {
            throw new ArgumentException(
                $"Input and output batch sizes must match. Input: {input.Input.Length}, Output: {destination.Length}.",
                nameof(destination));
        }

        if (input.Input.IsEmpty)
        {
            return ValueTask.CompletedTask;
        }

        if (input.ModelOutputs.Count == 0)
        {
            throw new InvalidOperationException("Multiple choice requires at least one model output.");
        }

        ReadOnlyTensorSpan<float> tensor = input.ModelOutputs.GetOutput(0);
        int rowCount = checked((int)tensor.Lengths[0]);
        if (rowCount != destination.Length)
        {
            throw new ArgumentException(
                $"The model produced {rowCount} result rows for an output batch of {destination.Length}.",
                nameof(destination));
        }

        Decode(input.Input, tensor, destination);
        return ValueTask.CompletedTask;
    }

    private void Decode(
        ReadOnlyMemory<TokenizedTextMultipleChoiceInput> input,
        ReadOnlyTensorSpan<float> tensor,
        Memory<ChoiceResult<TokenizedText>> output)
    {
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

    private ChoiceResult<TokenizedText> GetMultipleChoiceResult(
        TokenizedTextMultipleChoiceInput input,
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
}

[Obsolete("Use TextMultipleChoiceTensorization, ThenOnnxModel, and TextMultipleChoiceDecoding instead.")]
public sealed class TextMultipleChoicePipeline :
    IDestinationPipeline<ReadOnlyMemory<TokenizedTextMultipleChoiceInput>, Memory<ChoiceResult<TokenizedText>>>
{
    private readonly TextMultipleChoiceTensorization _tensorization;
    private readonly IPipeline<Tensor<long>[], TensorOutputs<float>> _modelPipeline;
    private readonly TextMultipleChoiceDecoding _decoding;

    public TextMultipleChoicePipeline(
        IPipeline<Tensor<long>[], TensorOutputs<float>> modelPipeline,
        TextMultipleChoiceOptions options)
    {
        _tensorization = new TextMultipleChoiceTensorization(options);
        _modelPipeline = modelPipeline;
        _decoding = new TextMultipleChoiceDecoding(options);
    }

    public async ValueTask<Memory<ChoiceResult<TokenizedText>>> ExecuteAsync(
        ReadOnlyMemory<TokenizedTextMultipleChoiceInput> input,
        CancellationToken cancellationToken = default)
    {
        Memory<ChoiceResult<TokenizedText>> output = new ChoiceResult<TokenizedText>[input.Length];
        await ExecuteAsync(input, output, cancellationToken);
        return output;
    }

    public async ValueTask ExecuteAsync(
        ReadOnlyMemory<TokenizedTextMultipleChoiceInput> input,
        Memory<ChoiceResult<TokenizedText>> destination,
        CancellationToken cancellationToken = default)
    {
        if (input.Length != destination.Length)
        {
            throw new ArgumentException("Input and output batch sizes must match.", nameof(destination));
        }

        if (input.IsEmpty)
        {
            return;
        }

        cancellationToken.ThrowIfCancellationRequested();
        Tensor<long>[] modelInput = await _tensorization.ExecuteAsync(input, cancellationToken);
        using TensorOutputs<float> modelOutput = await _modelPipeline.ExecuteAsync(modelInput, cancellationToken);
        await _decoding.ExecuteAsync((input, modelOutput), destination, cancellationToken);
    }
}
