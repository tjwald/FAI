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
- **Finite Pipelines**: ML tasks implement return-value `IPipeline<TInput, TOutput>`. Implement `IPreallocatingPipeline<TInput, TOutput>` only when output storage can be derived synchronously from metadata and execution can write a compatible destination.
- **DI Fluent API**: Start with `AddPipeline<TInput>()`, append typed pipelines with `Then<TOutput, TPipeline>()`, and use decorators to wrap the remainder of a chain.
- **Batch Policies**: Ordering, partitioning, routing, and scheduling compose around finite pipelines through indexed-batch operations.
