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
    IAllocatingStep<ReadOnlyMemory<Image<TPixel>>, Memory<ClassificationResult<TClassification, TFloat>>>
    where TPixel : unmanaged, IPixel<TPixel>
    where TFloat : unmanaged, IFloatingPointIeee754<TFloat>
{
    private readonly IImageProcessor<TPixel, TFloat> _imageProcessor;
    private readonly IBorrowedTensorProducer<Tensor<TFloat>[], TFloat> _modelStep;
    private readonly ClassificationOptions<TClassification> _options;
    private readonly ModelOutputConsumer _modelOutputConsumer;

    public ImageClassificationStep(
        IImageProcessor<TPixel, TFloat> imageProcessor,
        IBorrowedTensorProducer<Tensor<TFloat>[], TFloat> modelStep,
        ClassificationOptions<TClassification> options)
    {
        _imageProcessor = imageProcessor;
        _modelStep = modelStep;
        _options = options;
        _modelOutputConsumer = new ModelOutputConsumer(this);
    }

    public ValueTask<BatchLease<Memory<ClassificationResult<TClassification, TFloat>>>> RentOutputAsync(
        ReadOnlyMemory<Image<TPixel>> input,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        var output = new ClassificationResult<TClassification, TFloat>[input.Length];
        return ValueTask.FromResult(new BatchLease<Memory<ClassificationResult<TClassification, TFloat>>>(output));
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
        await _modelStep.ExecuteAsync(modelInput, output, _modelOutputConsumer, cancellationToken);
    }

    private sealed class ModelOutputConsumer(ImageClassificationStep<TPixel, TClassification, TFloat> owner) :
        IBorrowedTensorConsumer<TFloat, Memory<ClassificationResult<TClassification, TFloat>>>
    {
        public void Consume(
            ReadOnlyTensorSpan<TFloat> tensor,
            int outputIndex,
            Memory<ClassificationResult<TClassification, TFloat>> output)
        {
            if (outputIndex != 0)
            {
                return;
            }

            int rowCount = checked((int)tensor.Lengths[0]);
            if (rowCount != output.Length)
            {
                throw new InvalidOperationException(
                    $"The model produced {rowCount} result rows for an input batch of {output.Length}.");
            }

            int rowIndex = 0;
            foreach (ReadOnlyTensorSpan<TFloat> row in tensor.GetDimensionSpan(0))
            {
                output.Span[rowIndex] = owner._options.GetClassificationResult(row.AsSpan());
                rowIndex++;
            }
        }
    }
}
