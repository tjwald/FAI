# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Commands
- **Build**: `dotnet build FAI.slnx`
- **Lint**: `dotnet format` (part of pre-commit hooks)
- **Test**: `dotnet test` (Infrastructure initialized using `xunit.v3` and MTP in `test/` folder)
- **Post-Test**:
    - Always run `dotnet format` after tests pass.
    - Commit units of work after tests pass.

## Code Style (Non-Obvious)
- **Formatting**: 4 spaces, `LF` line endings, 160 chars max width.
- **Naming**: `_camelCase` for private/static fields; `PascalCase` for types, methods, and properties.
- **Modern C# (.NET 10 / C# 14)**:
    - Prefer collection expressions `[1, 2, 3]` over `new float[] { 1, 2, 3 }`.
    - Use `System.Threading.Lock` instead of `new object()` for locking.
- **Tensors**: Uses `System.Numerics.Tensors` (dotnet 9+ feature).

## Stability & Testing
- **Library Stability**: When working on tests, NEVER change the library code unless implementing a new feature (follow TDD).
- **Testing Style Guide**:
    - **Assertions**: Use explicit collection matching for ranges and outputs. Avoid partial assertions like `Assert.Single` when the full state can be verified.
    - **Mocks**: When testing decorators or schedulers, verify that exact input ranges reach the inner pipeline and preallocated slices receive the expected values.
    - **DI Testing**: Focus on verifying that the correct implementation types are resolved and that the component chain is assembled in the intended order.
    - **Collection Expressions**: Use `[1, 2, 3]` instead of `new int[] { 1, 2, 3 }` in all test code.

## Critical Patterns
- **Finite Pipelines**: ML tasks implement return-value `IPipeline<TInput, TOutput>`. Implement `IDestinationPipeline<TInput, TOutput>` when execution can write into a caller-supplied destination buffer without intermediate allocations.
- **Composition & Lifecycle**: Compose sequential steps with `AppendedPipeline.Create(prev, next)`. Branch inputs with `ForkPipeline`. Always dispose intermediate values using `PipelineOutputDisposer.DisposeAsync`.
- **Destination Adaptation**: Normalize pipelines using `pipeline.AsDestinationPipeline(outputBatch)`. Avoid runtime type-testing in hot execution loops.
- **Indexed Batch Traits**: Operations (`Count`, `Slice`, `Copy`, `Gather`, `Scatter`, `PermuteInPlace`) live in `IReadOnlyIndexedBatch<TBatch>` and `IWritableIndexedBatch<TBatch>`. Never pollute data types with batch traits.
- **Batch Policies**:
    - `PartitioningPipeline`: Slices input and destination contiguously. Never re-allocates per partition when a destination buffer is supplied.
    - `OrderingPipeline`: Gathers sorted inputs, executes inner pipeline, and calls `PermuteInPlace` to restore original order.
    - `RoutingPipeline`: Sorts non-contiguous routes into contiguous destination slices, executes pre-wrapped `IDestinationPipeline` targets, and calls `PermuteInPlace` once.
- **DI Fluent API**:
    - Start with `AddPipeline<TInput>()`, chain steps with `Then<TNext, TPipeline>()`, `ThenOnnxModel()`, or `.Fork()`.
    - Apply stage policies using `.Use(decorator)`. Decorators wrap the entire downstream chain. To constrain scope, nest via `.Then(inner => inner.Use(...).Then(...))`.
    - Register batch operations in DI with `services.AddMemoryBatch<T>()` and `services.AddTensorBatch<T>()`.
    - Resolve batch operations in decorators with `serviceProvider.GetRequiredWritableBatch<T>()`. NEVER use runtime reflection (`MakeGenericType` / `Activator.CreateInstance`).
