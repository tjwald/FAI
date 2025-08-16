using System.Numerics;
using System.Numerics.Tensors;
using FAI.Core.Abstractions;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace FAI.Vision.ImagePreProcessors;

public interface IImageProcessor<TPixel, TFloat> : IPreprocessor<Image<TPixel>, Tensor<TFloat>[], TFloat>
    where TPixel : unmanaged, IPixel<TPixel>
    where TFloat : IFloatingPointIeee754<TFloat>
{
    public Tensor<TFloat> Preprocess(Image<TPixel> image);
}