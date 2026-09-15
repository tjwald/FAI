# Vision Domain Architecture

This directory provides image preprocessing and computer vision inference pipelines using SixLabors.ImageSharp and `System.Numerics.Tensors`.

## 1. Execution Pipeline

Image inference is modeled as an end-to-end finite pipeline with native destination execution:

```mermaid
graph LR
    Images[ReadOnlyMemory&lt;Image&lt;TPixel&gt;&gt;] --> Prep[IImageProcessor<br/>Resize, Normalize, Channel Pack]
    Prep --> Tensors[Tensor&lt;TFloat&gt;[]<br/>NCHW / NHWC Tensors]
    Tensors --> Model[Model Pipeline<br/>ONNX Execution]
    Model --> Outputs[TensorOutputs&lt;TFloat&gt;]
    Outputs --> Decode[Synchronous Classification Decoding]
    Decode --> Dest[Memory&lt;ClassificationResult&gt;<br/>Caller-Owned Destination]
```

---

## 2. Image Preprocessing (`IImageProcessor<TPixel, TFloat>`)

```csharp
public interface IImageProcessor<TPixel, TFloat>
    where TPixel : unmanaged, IPixel<TPixel>
    where TFloat : IFloatingPointIeee754<TFloat>
{
    Tensor<TFloat>[] Preprocess(ReadOnlySpan<Image<TPixel>> images);
    Tensor<TFloat> Preprocess(Image<TPixel> image);
}
```

- Transforms batches of ImageSharp `Image<TPixel>` instances into dense input tensors.
- Handles resizing, color space conversions, channel normalization, and transposition (e.g. RGB HWC $\to$ CHW).

---

## 3. Destination Execution (`ImageClassificationPipeline`)

Implements `IDestinationPipeline<ReadOnlyMemory<Image<TPixel>>, Memory<ClassificationResult<TClassification, TFloat>>>`:

- **Caller-Supplied Destination (`ExecuteAsync(input, destination)`)**:
  - Validates cardinality match: `input.Length == destination.Length`.
  - Runs batch image preprocessing to produce model input tensors.
  - Asynchronously executes the model pipeline.
  - Synchronously decodes output logits directly into `destination.Span` without intermediate heap allocations.
- **Return-Value Path (`ExecuteAsync(input)`)**:
  - Allocates an array of `ClassificationResult<TClassification, TFloat>[input.Length]`.
  - Calls `ExecuteAsync(input, output)` and returns the populated buffer.
