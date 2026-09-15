---
name: fai-pipeline-design
description: "Use when designing, implementing, or composing finite ML pipelines in FAI. Covers IPipeline vs IDestinationPipeline, PipelineBuilder fluent API, Then, Fork, ThenOnnxModel, AppendedPipeline, and resource disposal."
---

# FAI Pipeline Design Skill

This skill guides agents when designing, authoring, or refactoring finite pipelines in the FAI repository.

## Core Mental Model

In FAI, every ML step transforms a **complete finite input value** into a **complete output value**:
- No streaming tokens, generators, or leaky iterators in hot inference paths.
- Execution operates on complete batches, tensor bundles, or domain results.
- 7X–14X speedups come from **zero-allocation destination execution**, cache-friendly memory reuse, and asynchronous unmanaged scope borrowing.

---

## 1. Choosing the Right Pipeline Interface

Always select the minimal appropriate interface:

```mermaid
decision-matrix
    "Can the component write into a caller-supplied buffer without intermediate allocations?"
    --> |Yes| IDestinationPipeline
    --> |No| IPipeline
```

### `IPipeline<in TInput, TOutput>`
Use when:
- Output dimensions or shapes cannot be predicted before execution (e.g. `TextTokenization`).
- The step returns unmanaged or runtime-borrowed objects (e.g. `TensorOutputs<float>` from ONNX).
- The step creates objects with dynamic cardinality.

```csharp
public sealed class MyTokenization : IPipeline<ReadOnlyMemory<string>, ReadOnlyMemory<TokenizedText>>
{
    public ValueTask<ReadOnlyMemory<TokenizedText>> ExecuteAsync(
        ReadOnlyMemory<string> input,
        CancellationToken cancellationToken = default)
    {
        // Compute and return complete value
    }
}
```

### `IDestinationPipeline<in TInput, TOutput>`
Use when:
- The component can write directly into caller-provided or sliced destination storage without allocating fresh heap memory.
- Standard for decoders (e.g. `ClassificationDecoding`), batch policies, and image preprocessors.

```csharp
public sealed class MyDecoder : IDestinationPipeline<TensorOutputs<float>, Memory<ClassificationResult>>
{
    // Return-value fallback: allocates once and delegates to destination write
    public async ValueTask<Memory<ClassificationResult>> ExecuteAsync(
        TensorOutputs<float> input,
        CancellationToken cancellationToken = default)
    {
        Memory<ClassificationResult> destination = new ClassificationResult[input.RowCount];
        await ExecuteAsync(input, destination, cancellationToken);
        return destination;
    }

    // Zero-allocation hot path
    public ValueTask ExecuteAsync(
        TensorOutputs<float> input,
        Memory<ClassificationResult> destination,
        CancellationToken cancellationToken = default)
    {
        ReadOnlyTensorSpan<float> tensor = input.GetOutput(0);
        // Write results directly into destination.Span
        return ValueTask.CompletedTask;
    }
}
```

---

## 2. Composing Pipelines with `PipelineBuilder`

Always start with `services.AddPipeline<TStart>()` and chain stages:

```csharp
services
    .AddPipeline<ReadOnlyMemory<string>>()
    .Then<ReadOnlyMemory<TokenizedText>, TextTokenization>()
    .UseTokenCountOrdering()
    .UseMaxPaddedTokensPartitioning()
    .Then<Tensor<long>[], TextTensorization>()
    .ThenOnnxModel()
    .Then<Memory<ClassificationResult<bool, float>>, ClassificationDecoding<bool>>()
    .Build();
```

### Rule 1: Type Checking at Compile Time
Each `.Then<TNext, TPipeline>()` changes the builder's current output type. Incompatible stage types fail at compile time.

### Rule 2: Single Builder Class
`PipelineBuilder<TStart, TCurrent>` is the single generic builder type. Do not introduce new builder classes or expose internal stage interfaces.

### Rule 3: Branching with `Fork`
When input data must be preserved alongside model output (e.g. multiple-choice tasks needing premise tokens alongside model scores):
```csharp
.Fork(inner => inner
    .Then<Tensor<long>[], MyTensorization>()
    .ThenOnnxModel())
// Downstream receives (TInput Input, TensorOutputs<float> Output)
```

### Rule 4: Scoping Decorators
`.Use(decorator)` wraps the entire remainder of the downstream chain. To limit a decorator's scope, nest the stages within a sub-pipeline:
```csharp
.Then(inner => inner
    .Use(myDecorator)
    .Then<StepA>()
    .Then<StepB>())
```

---

## 3. Resource Lifetime & Disposal Rules

1. **Intermediate Values**: Owned by the pipeline chain. Intermediate values that implement `IAsyncDisposable` or `IDisposable` are automatically disposed via `PipelineOutputDisposer.DisposeAsync`.
2. **Model Outputs (`TensorOutputs<T>`)**: Borrow unmanaged native engine handles (e.g. `OrtValue`). Never attempt to store `TensorOutputs<T>` beyond the scope of a decoder callback.
3. **Leased Buffers (`BatchLease<T>`)**: Always wrapped in a `using` block when renting from `ArrayPool<T>`:
   ```csharp
   using BatchLease<TInput> lease = inputBatch.Gather(input, indices);
   // use lease.Value
   ```
