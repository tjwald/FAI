# Testing Guide

FAI uses `xunit.v3` with Microsoft Testing Platform (MTP) for testing.

## Running Tests

All tests are located in the `test/` directory and use `net10.0`.

### Command Line

To run all tests:

```bash
dotnet test
```

To run tests for a specific project:

```bash
dotnet test test/FAI.Core.Tests/FAI.Core.Tests.csproj
```

### Visual Studio Code

You can use the built-in Test Explorer in VS Code to run and debug tests.

## Infrastructure

The testing infrastructure is centralized in `test/Directory.Build.props`, which includes:
- `xunit.v3`
- `Microsoft.NET.Test.Sdk`
- MTP runner enabled via `<UseMicrosoftTestingPlatformRunner>true</UseMicrosoftTestingPlatformRunner>`

## Project Coverage

The test suite covers the following areas:

- **Core**: [`test/FAI.Core.Tests`](test/FAI.Core.Tests) - Unit tests for core abstractions, batching logic, and tensor utilities.
- **DI (Dependency Injection)**: [`test/FAI.Core.Extensions.DI.Tests`](test/FAI.Core.Extensions.DI.Tests) - Verification of pipeline assembly and service registrations.
- **NLP**: [`test/FAI.NLP.Tests`](test/FAI.NLP.Tests) and [`test/FAI.NLP.Extensions.DI.Tests`](test/FAI.NLP.Extensions.DI.Tests) - Tokenization, NLP batching, and text-specific tasks.
- **Onnx**: [`test/FAI.Onnx.Tests`](test/FAI.Onnx.Tests) - ONNX model execution, device pools, and tensor utilities.
- **Evaluation**: [`test/FAI.Extensions.Evaluation.Tests`](test/FAI.Extensions.Evaluation.Tests) - Pipeline for batch evaluation of models against datasets.
- **Integration**: [`test/FAI.IntegrationTests`](test/FAI.IntegrationTests) - End-to-end verification of full pipeline assembly and execution using logical mocks.

## Integration Testing Strategy

Testing every possible permutation of pipeline components (Schedulers, Slicers, Executors) is neither feasible nor desirable. Instead, we follow a "Representative Combinations" strategy:

1. **Common Architectural Patterns**: We prioritize testing configurations used in production-like scenarios, such as `Background -> Partition -> Tokenizer -> Model`.
2. **Component Breadth**: Every major abstraction (e.g., `IPipelineBatchExecutor<TIn, TOut>`, `IBatchSchedular<TIn, TOut>`) must be included in at least one end-to-end integration test.
3. **Lifecycle & Concurrency**: We focus on cross-component state management, ensuring that data flows correctly through asynchronous boundaries (like `BackgroundPipelineBatchExecutor`) and partitioning logic.
4. **Logical Mocks**: We use manual implementation of `IModelExecutor` (e.g., `LogicalMockModelExecutor`) to verify the *library's* orchestration logic without the overhead or non-determinism of actual ONNX/PyTorch models.
