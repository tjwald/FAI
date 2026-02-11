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

## Critical Patterns
- **Middleware Chain**: `IPipelineBatchExecutor` follows a decorator/middleware pattern.
- **DI Fluent API**: Use `PipelineBuilder<TIn, TOut>` to assemble pipelines; executors are added in stack order (last added runs after previous).
- **Abstractions**: All ML tasks must implement `IInferenceSteps<TInput, TOutput>` or extend `InferenceSteps<...>`.
