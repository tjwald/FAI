# AGENTS.md

This file provides guidance to agents when working in Architect mode within this repository.

## Architectural Principles (Non-Obvious)
- **Extreme Performance**: The core goal is 7X-14X speedup over standard Python stacks. Every design decision must prioritize throughput and latency.
- **Pipeline Abstraction**: The library centers on [`IPipeline<TIn, TOut>`](src/FAI.Core/Abstractions.cs:88). It separates "what" to run (Inference Steps) from "how" to execute (Pipeline Batch Executors).
- **Batching Strategy**: Performance comes from specialized batching. Architects should consider new implementations of [`IPipelineBatchExecutor`](src/FAI.Core/Abstractions.cs:70) for specific hardware or data patterns (e.g., [`TokenCountSortingBatchExecutor`](src/FAI.NLP/PipelineBatchExecutors/TokenCountSortingBatchExecutor.cs)).
- **Hardware Agnostic**: Inference logic should be decoupled from the framework (ONNX, PyTorch, etc.) and hardware (CPU, GPU, OpenVino).

## Core Layout
- `FAI.Core`: Foundation interfaces and base execution logic.
- `FAI.NLP` / `FAI.Vision`: Domain-specific implementations (tokenizers, preprocessors).
- `FAI.Onnx`: Concrete model execution using ONNX Runtime.
- `*.Extensions.DI`: Fluent builders and ServiceCollection integration.
