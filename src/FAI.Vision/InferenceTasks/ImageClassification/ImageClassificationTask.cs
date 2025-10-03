using System.Numerics;
using System.Numerics.Tensors;
using FAI.Core.Abstractions;
using FAI.Core.Configurations.InferenceTasks;
using FAI.Core.InferenceTasks.Classification;
using FAI.Vision.ImagePreProcessors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace FAI.Vision.InferenceTasks.ImageClassification;

public class ImageClassification<TPixel, TClassification, TFloat>
    : ClassificationTask<Image<TPixel>, Tensor<TFloat>[], TFloat, TClassification, TFloat> where TFloat : unmanaged, IFloatingPointIeee754<TFloat>
    where TPixel : unmanaged, IPixel<TPixel>
{
    public ImageClassification(
        IImageProcessor<TPixel, TFloat> imageProcessor,
        IModelExecutor<TFloat, TFloat> modelExecutor,
        ClassificationOptions<TClassification> classificationOptions) : base(imageProcessor, modelExecutor, classificationOptions)
    {
    }
}
