using System.Numerics;
using System.Numerics.Tensors;
using FAI.Core;
using FAI.Core.Configurations.InferenceTasks;
using FAI.Core.ResultTypes;
using FAI.Core.Steps;
using FAI.Vision.ImagePreProcessors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace FAI.Vision.InferenceTasks.ImageClassification;

public sealed class ImageClassificationStep<TPixel, TClassification, TFloat> :
    IPreallocatingStep<ReadOnlyMemory<Image<TPixel>>, Memory<ClassificationResult<TClassification, TFloat>>>
    where TPixel : unmanaged, IPixel<TPixel>
    where TFloat : unmanaged, IFloatingPointIeee754<TFloat>
{
    private readonly IImageProcessor<TPixel, TFloat> _imageProcessor;
    private readonly IStep<Tensor<TFloat>[], TensorOutputs<TFloat>> _modelStep;
    private readonly ClassificationOptions<TClassification> _options;

    public ImageClassificationStep(
        IImageProcessor<TPixel, TFloat> imageProcessor,
        IStep<Tensor<TFloat>[], TensorOutputs<TFloat>> modelStep,
        ClassificationOptions<TClassification> options)
    {
        _imageProcessor = imageProcessor;
        _modelStep = modelStep;
        _options = options;
    }

    public bool TryAllocateOutput(
        ReadOnlyMemory<Image<TPixel>> input,
        out Memory<ClassificationResult<TClassification, TFloat>> output)
    {
        output = new ClassificationResult<TClassification, TFloat>[input.Length];
        return true;
    }

    public async ValueTask<Memory<ClassificationResult<TClassification, TFloat>>> ExecuteAsync(
        ReadOnlyMemory<Image<TPixel>> input,
        CancellationToken cancellationToken = default)
    {
        _ = TryAllocateOutput(input, out Memory<ClassificationResult<TClassification, TFloat>> output);
        await ExecuteAsync(input, output, cancellationToken);
        return output;
    }

    public async ValueTask ExecuteAsync(
        ReadOnlyMemory<Image<TPixel>> input,
        Memory<ClassificationResult<TClassification, TFloat>> output,
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

        Tensor<TFloat>[] modelInput = _imageProcessor.Preprocess(input.Span);
        using TensorOutputs<TFloat> modelOutput = await _modelStep.ExecuteAsync(modelInput, cancellationToken);
        ReadOnlyTensorSpan<TFloat> tensor = modelOutput.GetOutput(0);
        int rowCount = checked((int)tensor.Lengths[0]);
        if (rowCount != output.Length)
        {
            throw new InvalidOperationException(
                $"The model produced {rowCount} result rows for an input batch of {output.Length}.");
        }

        int rowIndex = 0;
        foreach (ReadOnlyTensorSpan<TFloat> row in tensor.GetDimensionSpan(0))
        {
            output.Span[rowIndex] = _options.GetClassificationResult(row.AsSpan());
            rowIndex++;
        }
    }
}
