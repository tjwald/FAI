using System.Numerics;
using System.Numerics.Tensors;
using FAI.Core;
using FAI.Core.Configurations.InferenceTasks;
using FAI.Core.Pipelines;
using FAI.Core.ResultTypes;
using FAI.Vision.ImagePreProcessors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace FAI.Vision.InferenceTasks.ImageClassification;

public sealed class ImageClassificationPipeline<TPixel, TClassification, TFloat> :
    IDestinationPipeline<ReadOnlyMemory<Image<TPixel>>, Memory<ClassificationResult<TClassification, TFloat>>>
    where TPixel : unmanaged, IPixel<TPixel>
    where TFloat : unmanaged, IFloatingPointIeee754<TFloat>
{
    private readonly IImageProcessor<TPixel, TFloat> _imageProcessor;
    private readonly IPipeline<Tensor<TFloat>[], TensorOutputs<TFloat>> _modelPipeline;
    private readonly ClassificationOptions<TClassification> _options;

    public ImageClassificationPipeline(
        IImageProcessor<TPixel, TFloat> imageProcessor,
        IPipeline<Tensor<TFloat>[], TensorOutputs<TFloat>> modelPipeline,
        ClassificationOptions<TClassification> options)
    {
        _imageProcessor = imageProcessor;
        _modelPipeline = modelPipeline;
        _options = options;
    }

    public async ValueTask<Memory<ClassificationResult<TClassification, TFloat>>> ExecuteAsync(
        ReadOnlyMemory<Image<TPixel>> input,
        CancellationToken cancellationToken = default)
    {
        Memory<ClassificationResult<TClassification, TFloat>> output = new ClassificationResult<TClassification, TFloat>[input.Length];
        await ExecuteAsync(input, output, cancellationToken);
        return output;
    }

    public async ValueTask ExecuteAsync(
        ReadOnlyMemory<Image<TPixel>> input,
        Memory<ClassificationResult<TClassification, TFloat>> destination,
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

        Tensor<TFloat>[] modelInput = _imageProcessor.Preprocess(input.Span);
        using TensorOutputs<TFloat> modelOutput = await _modelPipeline.ExecuteAsync(modelInput, cancellationToken);
        ReadOnlyTensorSpan<TFloat> tensor = modelOutput.GetOutput(0);
        int rowCount = checked((int)tensor.Lengths[0]);
        if (rowCount != destination.Length)
        {
            throw new InvalidOperationException(
                $"The model produced {rowCount} result rows for an input batch of {destination.Length}.");
        }

        int rowIndex = 0;
        foreach (ReadOnlyTensorSpan<TFloat> row in tensor.GetDimensionSpan(0))
        {
            destination.Span[rowIndex] = _options.GetClassificationResult(row.AsSpan());
            rowIndex++;
        }
    }
}
