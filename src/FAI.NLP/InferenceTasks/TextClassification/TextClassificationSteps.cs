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

public sealed class TextBatchEncodingStep : IStep<ReadOnlyMemory<TokenizedText>, Tensor<long>[]>
{
    private readonly PretrainedTokenizer _tokenizer;

    public TextBatchEncodingStep(PretrainedTokenizer tokenizer)
    {
        _tokenizer = tokenizer;
    }

    public ValueTask<Tensor<long>[]> ExecuteAsync(
        ReadOnlyMemory<TokenizedText> input,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (input.IsEmpty)
        {
            throw new ArgumentException("Cannot encode an empty text batch.", nameof(input));
        }

        return ValueTask.FromResult(Encode(input));
    }

    private Tensor<long>[] Encode(ReadOnlyMemory<TokenizedText> input)
    {
        var tokenizer = new TokenizerWrapper(_tokenizer);
        return tokenizer.Preprocess(input.Span).ToArray();
    }
}

public sealed class ClassificationDecodingStep<TClassification> :
    IPreallocatingStep<TensorOutputs<float>, Memory<ClassificationResult<TClassification, float>>>
{
    private readonly ClassificationOptions<TClassification> _options;

    public ClassificationDecodingStep(ClassificationOptions<TClassification> options)
    {
        _options = options;
    }

    public bool TryAllocateOutput(
        TensorOutputs<float> input,
        out Memory<ClassificationResult<TClassification, float>> output)
    {
        if (input.Count == 0)
        {
            output = default;
            return false;
        }

        int rowCount = checked((int)input.GetOutput(0).Lengths[0]);
        output = new ClassificationResult<TClassification, float>[rowCount];
        return true;
    }

    public async ValueTask<Memory<ClassificationResult<TClassification, float>>> ExecuteAsync(
        TensorOutputs<float> input,
        CancellationToken cancellationToken = default)
    {
        if (!TryAllocateOutput(input, out Memory<ClassificationResult<TClassification, float>> output))
        {
            throw new InvalidOperationException("Classification requires at least one model output.");
        }

        await ExecuteAsync(input, output, cancellationToken);
        return output;
    }

    public ValueTask ExecuteAsync(
        TensorOutputs<float> input,
        Memory<ClassificationResult<TClassification, float>> output,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        ReadOnlyTensorSpan<float> tensor = input.GetOutput(0);
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

        return ValueTask.CompletedTask;
    }
}
