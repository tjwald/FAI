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
- **DI Assembly**: Use `AddPipeline<TInput>()` and `Then<TOutput, TStep>()` to construct compile-time typed finite pipelines.
- **Decorator Scope**: Configure decorators on the `Then` stage they govern; decorators are declared outermost to innermost.
- **Inference Implementation**: Implement `IStep<TInput, TOutput>` or `IAllocatingStep<TInput, TOutput>` and mutate the supplied output.
