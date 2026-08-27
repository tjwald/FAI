using System.Numerics.Tensors;
using FAI.Core;
using FAI.Core.Configurations.InferenceTasks;
using FAI.Core.ResultTypes;
using FAI.Core.Steps;
using FAI.NLP.Tokenization;

namespace FAI.NLP.InferenceTasks.TextClassification;

internal sealed class TokenizerWrapper
{
    private readonly PretrainedTokenizer _tokenizer;

    public TokenizerWrapper(PretrainedTokenizer tokenizer)
    {
        _tokenizer = tokenizer;
    }

    public BatchTokenizedResult Preprocess(ReadOnlySpan<TokenizedText> input)
    {
        if (input[0].Tokens is null)
        {
            return _tokenizer.BatchTokenize(new TextView(input));
        }

        (Tensor<long> tokenization, Tensor<long> mask) = _tokenizer.BatchTokensToTensors(new TokensView(input));

        return new BatchTokenizedResult(tokenization, mask);
    }
}

public sealed class TextBatchEncodingStep : IAllocatingStep<ReadOnlyMemory<TokenizedText>, Tensor<long>[]>
{
    private readonly PretrainedTokenizer _tokenizer;

    public TextBatchEncodingStep(PretrainedTokenizer tokenizer)
    {
        _tokenizer = tokenizer;
    }

    public ValueTask<BatchLease<Tensor<long>[]>> RentOutputAsync(
        ReadOnlyMemory<TokenizedText> input,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        return ValueTask.FromResult(new BatchLease<Tensor<long>[]>(new Tensor<long>[2]));
    }

    public ValueTask<BatchLease<Tensor<long>[]>> ExecuteAsync(
        ReadOnlyMemory<TokenizedText> input,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        return ValueTask.FromResult(new BatchLease<Tensor<long>[]>(Encode(input)));
    }

    public ValueTask ExecuteAsync(
        ReadOnlyMemory<TokenizedText> input,
        Tensor<long>[] output,
        CancellationToken cancellationToken = default)
    {
        if (output.Length != 2)
        {
            throw new ArgumentException("Text encoding requires token and attention-mask destinations.", nameof(output));
        }

        if (input.IsEmpty)
        {
            throw new ArgumentException("Cannot encode an empty text batch.", nameof(input));
        }

        cancellationToken.ThrowIfCancellationRequested();
        Tensor<long>[] encoded = Encode(input);
        encoded.CopyTo(output, 0);
        return ValueTask.CompletedTask;
    }

    private Tensor<long>[] Encode(ReadOnlyMemory<TokenizedText> input)
    {
        var tokenizer = new TokenizerWrapper(_tokenizer);
        return tokenizer.Preprocess(input.Span).ToArray();
    }
}

public sealed class ClassificationDecodingStep<TClassification> :
    IBorrowedTensorConsumer<float, Memory<ClassificationResult<TClassification, float>>>
{
    private readonly ClassificationOptions<TClassification> _options;

    public ClassificationDecodingStep(ClassificationOptions<TClassification> options)
    {
        _options = options;
    }

    public void Consume(
        ReadOnlyTensorSpan<float> tensor,
        int outputIndex,
        Memory<ClassificationResult<TClassification, float>> output)
    {
        if (outputIndex != 0)
        {
            return;
        }

        int rowCount = checked((int)tensor.Lengths[0]);
        if (rowCount != output.Length)
        {
            throw new ArgumentException(
                $"The model produced {rowCount} result rows for an output batch of {output.Length}.",
                nameof(output));
        }

        Span<ClassificationResult<TClassification, float>> outputSpan = output.Span;
        int rowIndex = 0;
        foreach (ReadOnlyTensorSpan<float> row in tensor.GetDimensionSpan(0))
        {
            outputSpan[rowIndex] = _options.GetClassificationResult(row.AsSpan());
            rowIndex++;
        }
    }
}
