using System.Numerics.Tensors;
using FAI.Core;
using FAI.Core.Configurations.InferenceTasks;
using FAI.Core.Pipelines;
using FAI.Core.ResultTypes;
using FAI.NLP.Tokenization;

namespace FAI.NLP.InferenceTasks.TextClassification;

public sealed class TextTensorization : IPipeline<ReadOnlyMemory<TokenizedText>, Tensor<long>[]>
{
    private readonly PretrainedTokenizer _tokenizer;

    public TextTensorization(PretrainedTokenizer tokenizer)
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
        return _tokenizer.BatchTokensToTensors(new TokensView(input.Span)).ToArray();
    }
}

public sealed class ClassificationDecoding<TClassification> :
    IPreallocatingPipeline<TensorOutputs<float>, Memory<ClassificationResult<TClassification, float>>>
{
    private readonly ClassificationOptions<TClassification> _options;

    public ClassificationDecoding(ClassificationOptions<TClassification> options)
    {
        _options = options;
    }

    public async ValueTask<Memory<ClassificationResult<TClassification, float>>> ExecuteAsync(
        TensorOutputs<float> input,
        CancellationToken cancellationToken = default)
    {
        if (input.Count == 0)
        {
            throw new InvalidOperationException("Classification requires at least one model output.");
        }

        int rowCount = checked((int)input.GetOutput(0).Lengths[0]);
        Memory<ClassificationResult<TClassification, float>> output = new ClassificationResult<TClassification, float>[rowCount];
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
