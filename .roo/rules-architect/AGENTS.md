# AGENTS.md

This file provides guidance to agents when working in Architect mode within this repository.

## Architectural Principles (Non-Obvious)
- **Extreme Performance**: The core goal is 7X-14X speedup over standard Python stacks. Every design decision must prioritize throughput and latency.
- **Pipeline Abstraction**: The library centers on return-value `IPipeline<TInput, TOutput>`. Destination buffer execution uses `IDestinationPipeline<TInput, TOutput>` for zero-allocation writes into caller-owned buffers.
- **Composition & Lifecycle**: Pipelines compose with `AppendedPipeline.Create(prev, next)` and branch with `ForkPipeline`. Intermediate values are asynchronously disposed via `PipelineOutputDisposer.DisposeAsync`.
- **Destination Normalization**: Normalize inner pipelines to `IDestinationPipeline` once at construction via `inner.AsDestinationPipeline(outputBatch)`. Avoid runtime type-branching in execution loops.
- **Batching Strategy**: Performance comes from composable ordering, partitioning, routing, and scheduling policies over external indexed batch traits (`IReadOnlyIndexedBatch`, `IWritableIndexedBatch`). Domain packages should add policies rather than parallel execution abstractions.
- **Reflection-Free DI**: Batch operations are registered as typed services in DI (`services.AddMemoryBatch<T>()`, `services.AddTensorBatch<T>()`) and resolved via `serviceProvider.GetRequiredWritableBatch<T>()`. Never use runtime reflection (`MakeGenericType` / `Activator.CreateInstance`).
- **Hardware Agnostic**: Inference logic is decoupled from runtime backends and hardware (CPU, GPU, OpenVino). Model outputs borrow unmanaged buffers as `TensorOutputs<T>` for synchronous decoding without managed heap allocation.

## Core Layout
- `FAI.Core`: Foundation interfaces, finite pipeline engine, and indexed batch traits/policies.
- `FAI.Core.Extensions.DI`: Unified `PipelineBuilder<TStart, TCurrent>` fluent API and DI integration.
- `FAI.NLP` / `FAI.Vision`: Domain-specific implementations (tokenizers, preprocessors, decoders).
- `FAI.Onnx`: Concrete model execution using ONNX Runtime.
- `FAI.Extensions.Evaluation`: Decoupled streaming dataset evaluation pipeline with OpenTelemetry instrumentation.
