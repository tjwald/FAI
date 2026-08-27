# Finite pipelines

Finite pipelines are composed from steps that transform one complete input value into a caller-provided output value.

```csharp
public interface IStep<TInput, TOutput>
{
    ValueTask ExecuteAsync(
        TInput input,
        TOutput output,
        CancellationToken cancellationToken = default);
}
```

`TInput` and `TOutput` represent complete batches. A step can therefore use `ReadOnlyMemory<T>`, `Memory<T>`, `Tensor<T>`, or a model-specific tensor bundle without requiring one output object per input item.

## Output allocation

Steps that can produce intermediate values implement `IAllocatingStep<TInput, TOutput>`. The producing step owns its output shape and allocator selection, preventing a pipeline builder from pairing it with an incompatible output factory.

```csharp
public interface IAllocatingStep<TInput, TOutput> : IStep<TInput, TOutput>
{
    ValueTask<BatchLease<TOutput>> RentOutputAsync(
        TInput input,
        CancellationToken cancellationToken = default);

        ValueTask<BatchLease<TOutput>> ExecuteAsync(
        TInput input,
        CancellationToken cancellationToken = default);
}
```

    The direct `IStep.ExecuteAsync` path does not allocate an output. Pipeline composition uses the atomic allocating overload for intermediate links, allowing a step to prepare its input-dependent state once. `RentOutputAsync` remains available when a caller must separate allocation from execution. The final output is always supplied by the caller. The allocating overload returns a lease that the caller must dispose.

## Composition

`AddPipeline<TInput>()` starts a typed pipeline. Every `Then<TOutput, TStep>()` changes the builder's current type, so incompatible stages fail at compile time.

```csharp
services
    .AddPipeline<ReadOnlyMemory<string>>()
    .Then<TokenBatch, TokenizeStep>()
    .Then<Tensor<float>, EmbeddingStep>(stage => stage
        .Use<TokenCountOrderingStep>()
        .UseBatchPartitioning<TokenBatchOperations, TensorBatchOperations<float>>())
    .Then<Tensor<float>, NormalizeStep>()
    .Build("embeddings");
```

This registers one keyed `IStep<ReadOnlyMemory<string>, Tensor<float>>`. `Then` does not register output factories: every `TStep` must implement `IAllocatingStep` and provide its own intermediate allocation.

The builder compiles stages into a continuation-based chain. This allows each step to rent its output only after its immediate input exists. Intermediate leases are disposed immediately after the consuming stage completes.

`Use` wraps only the stage passed to that `Then` call. Policies are applied in declaration order from outermost to innermost. A step implementation does not call the next step; only pipeline chains and policy decorators hold and invoke an inner step.

## Nested pipelines

The pipeline-building `Then` overload turns another typed pipeline into one stage of the current pipeline. Decorators on that stage wrap the complete inner chain rather than one of its individual steps.

```csharp
services
    .AddPipeline<ReadOnlyMemory<TokenizedText>>()
    .Then<Memory<ClassificationResult<bool, float>>>(
        pipeline => pipeline
            .Then<Tensor<long>[], TextBatchEncodingStep>()
            .ThenBorrowed<float, Memory<ClassificationResult<bool, float>>>(
                services => services.GetRequiredService<IBorrowedTensorProducer<Tensor<long>[], float>>(),
                services => services.GetRequiredService<ClassificationDecodingStep<bool>>(),
                (_, input, _) => ValueTask.FromResult(
                    new BatchLease<Memory<ClassificationResult<bool, float>>>(
                        new ClassificationResult<bool, float>[input[0].Lengths[0]]))),
        (_, input, _) => ValueTask.FromResult(
            new BatchLease<Memory<ClassificationResult<bool, float>>>(
                new ClassificationResult<bool, float>[input.Length])),
        stage => stage
            .UseTokenizingStep()
            .UseTokenCountOrderingStep()
            .UseMaxPaddedTokensPartitioningStep())
    .Build();
```

The nested stage requires an explicit endpoint allocator. Its final step normally allocates from its immediate input, but that input does not exist until the earlier nested steps have executed. The endpoint allocator instead describes how the enclosing pipeline allocates output directly from the enclosing input without executing the nested pipeline twice.

For classification, the effective execution order is tokenization, token-count ordering, token-budget partitioning through the configured scheduler, tensor encoding, model execution, and logits decoding. Output order is restored only after the complete nested pipeline has finished.

## Indexed batch policies

Ordering, partitioning, and routing require first-axis batch semantics. These semantics use static abstract traits rather than injected operation services:

- `ReadOnlyMemoryBatchOperations<T>` gathers and slices memory inputs.
- `MemoryBatchOperations<T>` rents, slices, scatters, and permutes memory outputs.
- `TensorBatchOperations<T>` performs the same operations over axis zero of `Tensor<T>`.

Contiguous tensor slices are shared views of the source tensor. Partitioning can therefore write directly into the corresponding region of the caller's output. Non-contiguous ordering and routing require gathered temporary storage unless the underlying model accepts indexed inputs.

`PartitioningStep` forwards matching input and output views to its inner step. `OrderingStep` gathers inputs in the selected order, invokes its inner step, and restores caller output order. `RoutingStep` gathers each route, invokes its selected target, and scatters results to the original row positions.

Domain packages should expose concise extensions over these generic policies. For example, NLP can provide token-budget partitioning and token-count ordering without introducing another execution abstraction.

## Borrowed model output

Runtime-owned tensors should be decoded while the runtime allocation is alive. `IBorrowedTensorProducer<TInput, TElement>` invokes an `IBorrowedTensorConsumer<TElement, TOutput>` synchronously for each output tensor. The consumer receives a `ReadOnlyTensorSpan<TElement>` and must not retain it after `Consume` returns.

`ThenBorrowed` composes a producer and consumer into a normal owned-output pipeline stage. ONNX keeps each `OrtValue` alive through the consumer call, so decoding does not allocate or copy a managed logits tensor. Domain packages depend only on the Core borrowed contracts and never on ONNX types.

Use `ModelExecutorFactory.CreateBorrowedModelStep` for normal model-and-decode pipelines. `CreateMaterializingModelStep` is the explicit adapter for consumers that require owned `Tensor<float>[]` output.

## Ownership

Inputs and caller-provided outputs must remain valid until `ExecuteAsync` completes. A `BatchLease<T>` owns only the value returned by an allocating execution or `RentOutputAsync`; disposing it returns pooled resources. Tensor views returned by `TensorBatchOperations<T>.Slice` share their source storage and do not own it. Borrowed tensor spans are valid only during their synchronous consumer call.
