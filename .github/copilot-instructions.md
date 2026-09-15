# GitHub Copilot Instructions for FAI

You are working in the FAI repository, a high-performance .NET 10 / C# 14 machine learning inference infrastructure library designed for 7X–14X speedups over standard Python inference stacks.

## Core Architecture Principles

1. **Finite Pipelines (`IPipeline<TInput, TOutput>`)**:
   - Every ML step transforms a complete finite input value into a complete output value via `ValueTask<TOutput> ExecuteAsync(TInput input, CancellationToken ct = default)`.
   - Returned values belong to the caller.
   - Intermediate values are owned by the pipeline chain and must be disposed via `PipelineOutputDisposer.DisposeAsync`.

2. **Destination Execution (`IDestinationPipeline<TInput, TOutput>`)**:
   - When a component can write directly into caller-supplied or sliced buffers without intermediate allocations, implement `IDestinationPipeline<TInput, TOutput>` via `ValueTask ExecuteAsync(TInput input, TOutput destination, CancellationToken ct = default)`.
   - Normalize inner pipelines using `pipeline.AsDestinationPipeline(outputBatch)` during construction to eliminate per-batch type checking.

3. **External Indexed Batch Traits**:
   - Batch traits (`Count`, `Slice`, `Copy`, `Gather`, `Scatter`, `PermuteInPlace`) live in `IReadOnlyIndexedBatch<TBatch>` and `IWritableIndexedBatch<TBatch>`.
   - Never implement batch interfaces on data models. Use built-in operations for `Memory<T>`, `ReadOnlyMemory<T>`, and `System.Numerics.Tensors.Tensor<T>`.

4. **Batch Policies**:
   - `PartitioningPipeline`: Contiguously slices input and destination. Zero-allocation execution when a destination is supplied.
   - `OrderingPipeline`: Gathers sorted inputs, executes inner pipeline, and calls `PermuteInPlace` to restore original order.
   - `RoutingPipeline`: Sorts non-contiguous routes into contiguous destination slices, executes pre-wrapped `IDestinationPipeline` targets, and calls `PermuteInPlace` once.

5. **Fluent DI API (`PipelineBuilder<TStart, TCurrent>`)**:
   - Single unified generic builder driven by internal `IStage` state machine (`InitialStage`, `ComposedStage`, `DecoratedStage`).
   - Chain steps with `.Then<TNext, TPipeline>()`, `.ThenOnnxModel()`, or `.Fork(...)`.
   - Apply stage policies using `.Use(decorator)`. Decorators wrap the entire downstream chain. To limit scope, nest via `.Then(inner => inner.Use(...).Then(...))`.
   - Register batch operations in DI with `services.AddMemoryBatch<T>()` and `services.AddTensorBatch<T>()`.
   - Resolve batch operations in decorators with `serviceProvider.GetRequiredWritableBatch<T>()`. NEVER use runtime reflection (`MakeGenericType` / `Activator.CreateInstance`).

6. **Borrowing Model Outputs (`TensorOutputs<T>`)**:
   - Models return borrowed native handles wrapped in Core's `TensorOutputs<T>`.
   - Decoders inspect `ReadOnlyTensorSpan<T>` synchronously without managed heap allocation.

## Coding Style & Tooling Rules
- **Formatting**: 4 spaces, `LF` line endings, max 160 characters width.
- **Modern C#**:
  - Use collection expressions `[1, 2, 3]` instead of `new int[] { 1, 2, 3 }`.
  - Use `System.Threading.Lock` instead of `new object()`.
- **CLI Commands**:
  - Build: `dotnet build FAI.slnx`
  - Lint: `dotnet format`
  - Test: `dotnet test`
  - Post-Test: Always run `dotnet format` after test runs.
