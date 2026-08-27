# AGENTS.md

This file provides guidance to agents when working in Architect mode within this repository.

## Architectural Principles (Non-Obvious)
- **Extreme Performance**: The core goal is 7X-14X speedup over standard Python stacks. Every design decision must prioritize throughput and latency.
- **Step Abstraction**: The library centers on finite `IStep<TInput, TOutput>` and `IAllocatingStep<TInput, TOutput>` contracts. Each step transforms one complete value and writes into caller-owned output.
- **Batching Strategy**: Performance comes from composable ordering, partitioning, routing, and scheduling policies over indexed batch traits. Domain packages should add policies rather than parallel execution abstractions.
- **Hardware Agnostic**: Inference logic should be decoupled from the framework (ONNX, PyTorch, etc.) and hardware (CPU, GPU, OpenVino).

## Core Layout
- `FAI.Core`: Foundation interfaces and base execution logic.
- `FAI.NLP` / `FAI.Vision`: Domain-specific implementations (tokenizers, preprocessors).
- `FAI.Onnx`: Concrete model execution using ONNX Runtime.
- `*.Extensions.DI`: Fluent builders and ServiceCollection integration.
