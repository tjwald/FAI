using System.Numerics;
using System.Numerics.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace FAI.Vision.ImagePreProcessors;

public interface IImageProcessor<TPixel, TFloat>
    where TPixel : unmanaged, IPixel<TPixel>
    where TFloat : IFloatingPointIeee754<TFloat>
{
    Tensor<TFloat>[] Preprocess(ReadOnlySpan<Image<TPixel>> images);

    Tensor<TFloat> Preprocess(Image<TPixel> image);
}
