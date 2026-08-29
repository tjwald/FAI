# Finite pipelines

Finite pipelines compose components that transform one complete value into another.

```csharp
public interface IPipeline<in TInput, TOutput>
{
    ValueTask<TOutput> ExecuteAsync(
        TInput input,
        CancellationToken cancellationToken = default);
}
```

`TInput` and `TOutput` may each represent a complete batch, tensor bundle, or runtime-owned model result. A returned value belongs to the caller. During composition, the chain owns intermediate values and disposes an `IAsyncDisposable` or `IDisposable` intermediate only after the complete downstream asynchronous operation finishes.

## Conditional preallocation

Preallocation is an optional capability, not the base execution contract.

```csharp
public interface IPreallocatingPipeline<in TInput, TOutput> : IPipeline<TInput, TOutput>
{
    bool TryAllocateOutput(TInput input, out TOutput output);

    ValueTask ExecuteAsync(
        TInput input,
        TOutput output,
        CancellationToken cancellationToken = default);
}
```

Interface presence means a pipeline understands destination execution. `TryAllocateOutput` determines whether metadata is sufficient for a particular input. It returns storage only and must not perform inference, tokenization, I/O, probing, or substantive work that execution would repeat. It must not encode hidden execution state into the destination.

A supplied compatible destination remains valid for `ExecuteAsync(input, output)` even when `TryAllocateOutput(input, out _)` returns `false`. Invalid metadata and allocation failures throw; `false` means only that this invocation cannot be preallocated from available metadata.

## Composition

`AddPipeline<TInput>()` starts a typed pipeline. Each `Then<TOutput, TPipeline>()` changes the current type, so incompatible components fail at compile time.

```csharp
services
    .AddPipeline<ReadOnlyMemory<string>>()
    .Then<ReadOnlyMemory<TokenizedText>, TextTokenization>()
    .UseTokenCountOrdering()
    .UseMaxPaddedTokensPartitioning()
    .Then<Tensor<long>[], TextTensorization>()
    .Then<TensorOutputs<float>>(services =>
        services.GetRequiredService<IPipeline<Tensor<long>[], TensorOutputs<float>>>() )
    .Then<Memory<ClassificationResult<bool, float>>, ClassificationDecoding<bool>>()
    .WithOutputAllocation((input, out output) =>
    {
        output = new ClassificationResult<bool, float>[input.Length];
        return true;
    })
    .Build();
```

Nested `Then` is an ordinary typed pipeline and needs no allocator for normal return-value execution. A type-changing pipeline may optionally declare a synchronous endpoint allocator with `WithOutputAllocation` when its final component supports destination execution but cannot derive final storage directly from the pipeline's starting input. The allocator follows the same storage-only Try contract and lets partition decorators preallocate one complete output.

`Use` wraps the complete remainder of the chain. Decorators execute in declaration order. To limit a decorator's scope, create that scope with nested `Then(pipeline => pipeline.Use(...).Then(...))`.

For classification, execution order is raw text tokenization, token-count ordering, token-budget partitioning, scheduled tensorization, model execution, decoding, and restoration of original order. `TextTokenization` changes raw strings into immutable `TokenizedText` values. `TextTensorization` only pads those token sequences and creates model input tensors; it never tokenizes or mutates text.

## Batch capabilities

Batch structure belongs to operations traits rather than `TInput` or `TOutput`. This supports external values such as arrays, `Memory<T>`, and `Tensor<T>` without requiring those types to implement FAI interfaces.

The relevant operations are independent capabilities:

- Cardinality
- Contiguous slicing
- Indexed gathering
- Aggregate allocation
- Scattering
- Permutation

Ordering requires indexed gathering and output permutation. Partitioning requires contiguous input slicing. Routing gathers non-contiguous inputs; its preallocated path writes route results into contiguous output slices and performs one final permutation, while its fallback scatters returned route outputs.

## Partition preallocation

Partitioning selects its strategy from three capabilities:

1. The input operations support contiguous slicing.
2. The output operations support matching destination slices.
3. The inner pipeline implements `IPreallocatingPipeline<TInput, TOutput>` and can allocate for the full input.

When all are available, partitioning calls `TryAllocateOutput` once with the complete input, slices the input and destination by matching ranges, and schedules writes into disjoint output slices. It never allocates once per partition.

If preallocation returns `false`, partitioning executes each partition through the ordinary return-value path, allocates an aggregate shaped from an actual partition result, and scatters each result into place.

Structural capabilities are selected when generic decorators are configured and the inner step interface is cached when DI constructs the pipeline. Runtime checks cover per-input preallocation eligibility, cardinality, shape compatibility, ranges, cancellation, and disposal.

## Runtime-owned model outputs

`TensorOutputs<T>` is a Core-owned disposable output value. ONNX implements it with live `OrtValue` instances and returns it from a normal model pipeline:

```csharp
IPipeline<Tensor<long>[], TensorOutputs<float>>
```

A decoder obtains `ReadOnlyTensorSpan<T>` views synchronously from the live scope. The scope can remain alive across awaits, while C# prevents a ref-struct span from crossing an await. The composed chain disposes the scope after decoding finishes, including failure and cancellation paths.

Normal classification therefore performs no managed logits materialization. A consumer that needs owned logits must copy them explicitly while the scope is alive.

## Capability examples

- Tokenization is an explicit type-changing, return-value-only pipeline because dimensions depend on tokenization work.
- Text tensorization is return-value-only when shape discovery would repeat tensor construction work.
- Static-shape model backends may implement preallocation only when they can bind supplied storage directly.
- Dynamic-output models remain return-value-only for inputs whose output shape is not metadata-predictable.
- Classification, multiple-choice, and image decoders can preallocate result memory from output cardinality and write directly into slices.

Tensor and memory slices share source storage and do not own it. Gathered pooled inputs are local decorator implementation details and must be disposed before the decorator returns.
