# AGENTS.md

This file provides guidance to agents when working in Code mode within this repository.

## Performance & Memory (CRITICAL)
- **Zero-Allocation**: Aim for zero-allocation in the hot path. Use `Span<T>`, `ReadOnlySpan<T>`, `Memory<T>`, and `ReadOnlyMemory<T>` to avoid copying data.
- **Tensors**: Always use `System.Numerics.Tensors`. Check [`TensorExtensions.cs`](src/FAI.Core/TensorExtensions.cs) for optimized operations.
- **Pooling**: Use `BatchLease<T>` for owned intermediate values and pools for expensive runtime objects.
- **Concurrency**: Use `SemaphoreSlim` for throttling and `Channel<T>` for producer/consumer patterns to manage throughput without blocking.

## Coding Rules (Non-Obvious)
- **Project Commands**:
    - **Build**: `dotnet build FAI.slnx`
    - **Lint**: `dotnet format`
    - **Test**: `dotnet test`
    - **Post-Test**:
        - Always run `dotnet format` after tests pass.
        - Commit units of work after tests pass.
- **Modern C# (.NET 10 / C# 14)**:
    - Prefer collection expressions `[1, 2, 3]` over `new float[] { 1, 2, 3 }`.
    - Use `System.Threading.Lock` instead of `new object()` for locking.
- **Stability**: When working on tests, NEVER change the library code unless implementing a new feature (follow TDD).
- **DI Assembly**: Use `AddPipeline<TInput>()`, chain steps with `Then<TNext, TPipeline>()`, `ThenOnnxModel()`, or `Fork(...)`.
- **Decorator Scope**: Configure decorators using `.Use(...)`; decorators wrap the complete remainder of the chain. To constrain scope, nest via `.Then(inner => inner.Use(...).Then(...))`.
- **Inference Implementation**: Implement return-value `IPipeline<TInput, TOutput>`. Add `IDestinationPipeline<TInput, TOutput>` when execution can write into a caller-supplied destination buffer without intermediate allocations.
- **Batch Operations**: Register batch operations in DI via `services.AddMemoryBatch<T>()` and `services.AddTensorBatch<T>()`. Resolve in decorators via `serviceProvider.GetRequiredWritableBatch<T>()`. NEVER use runtime reflection.
