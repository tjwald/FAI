# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Commands
- **Build**: `dotnet build FAI.slnx`
- **Lint**: `dotnet format` (part of pre-commit hooks)
- **Test**: `dotnet test` (Current coverage is 0 - add tests to new `test/` folder)

## Code Style (Non-Obvious)
- **Formatting**: 4 spaces, `LF` line endings, 160 chars max width.
- **Naming**: `_camelCase` for private/static fields; `PascalCase` for types, methods, and properties.
- **Modern C#**: Uses `.slnx` solution format. Target is `net10.0`.
- **Tensors**: Uses `System.Numerics.Tensors` (dotnet 9+ feature).

## Critical Patterns
- **Middleware Chain**: `IPipelineBatchExecutor` follows a decorator/middleware pattern.
- **DI Fluent API**: Use `PipelineBuilder<TIn, TOut>` to assemble pipelines; executors are added in stack order (last added runs after previous).
- **Abstractions**: All ML tasks must implement `IInferenceSteps<TInput, TOutput>` or extend `InferenceSteps<...>`.
