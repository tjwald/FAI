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

## Destination execution

Destination execution is an optional capability, not the base execution contract.

```csharp
public interface IDestinationPipeline<in TInput, TOutput> : IPipeline<TInput, TOutput>
{
    ValueTask ExecuteAsync(
        TInput input,
        TOutput destination,
        CancellationToken cancellationToken = default);
}
```

Interface presence means a pipeline understands destination execution. When a caller or decorator supplies an allocated destination buffer (e.g., sliced from a contiguous output buffer or pooled), `ExecuteAsync(input, destination)` writes results directly into that buffer without intermediate heap allocations.

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
    .Build();
```

`Use` wraps the complete remainder of the chain. Decorators execute in declaration order. To limit a decorator's scope, create that scope with nested `Then(pipeline => pipeline.Use(...).Then(...))`.

For classification, execution order is raw text tokenization, token-count ordering, token-budget partitioning, scheduled tensorization, model execution, decoding, and restoration of original order. `TextTokenization` changes raw strings into immutable `TokenizedText` values. `TextTensorization` only pads those token sequences and creates model input tensors; it never tokenizes or mutates text.

## Batch capabilities

Batch structure belongs to operations traits rather than `TInput` or `TOutput`. This supports external values such as arrays, `Memory<T>`, and `Tensor<T>` without requiring those types to implement FAI interfaces.

The relevant operations are independent capabilities:

- Cardinality
- Contiguous slicing
- Contiguous copying
- Indexed gathering
- Aggregate allocation
- Scattering
- Permutation

Ordering requires indexed gathering and output permutation. Partitioning requires contiguous input slicing. Routing gathers non-contiguous inputs; its destination path writes route results into contiguous output slices and performs one final permutation, while its fallback scatters returned route outputs.

## Partitioning execution

When called via `ExecuteAsync(input, destination)` with a caller-supplied destination, partitioning slices both the input and destination by matching ranges:

```csharp
await _destinationInner.ExecuteAsync(partitionInput, _outputBatch.Slice(destination, range), token);
```

When called via the return-value path `ExecuteAsync(input)`, partitioning executes the partitions through the inner pipeline, allocates an aggregate shaped from an actual partition result via `_outputBatch.AllocateLike(...)`, and copies each partition result into its slice.

Structural capabilities are selected when generic decorators are configured and the inner step interface is cached when DI constructs the pipeline. Runtime checks cover cardinality, shape compatibility, ranges, cancellation, and disposal.

## Routing execution

Routing groups non-contiguous input indices by target. When writing into a destination buffer:
1. Each target writes into its assigned contiguous slice of the destination buffer.
2. If a target implements `IDestinationPipeline`, it writes directly into that slice with zero intermediate allocation.
3. Once all routes complete, `PermuteInPlace` restores elements to their original batch positions in a single pass.

When no destination buffer is provided, routing executes each target, allocates an aggregate output using `_outputBatch.AllocateLike(...)`, and scatters each target's results into their respective batch indices.

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
