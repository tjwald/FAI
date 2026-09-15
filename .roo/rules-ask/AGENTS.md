# AGENTS.md

This file provides guidance to agents when working in Ask mode within this repository.

## Documentation Rules (Non-Obvious Only)
- **Performance Benchmarks**: Canonical performance gains (7X-14X) are documented in [`README.md`](README.md:20) and compared against standard Python stacks in the [`Examples/`](Examples/) directory.
- **Design Context**: High-level architecture and the motivation for the library are found in [`docs/high-level-design.md`](docs/high-level-design.md), [`docs/finite-pipelines.md`](docs/finite-pipelines.md), and [`docs/architecture-decisions.md`](docs/architecture-decisions.md).
- **Subsystem Architectures**: Each major subsystem maintains a localized `ARCHITECTURE.md` (e.g., [`src/FAI.Core/Pipelines/ARCHITECTURE.md`](src/FAI.Core/Pipelines/ARCHITECTURE.md), [`src/FAI.Core.Extensions.DI/ARCHITECTURE.md`](src/FAI.Core.Extensions.DI/ARCHITECTURE.md), [`src/FAI.NLP/ARCHITECTURE.md`](src/FAI.NLP/ARCHITECTURE.md), [`src/FAI.Onnx/ARCHITECTURE.md`](src/FAI.Onnx/ARCHITECTURE.md), [`src/FAI.Vision/ARCHITECTURE.md`](src/FAI.Vision/ARCHITECTURE.md), [`src/FAI.Extensions.Evaluation/ARCHITECTURE.md`](src/FAI.Extensions.Evaluation/ARCHITECTURE.md)).
- **Core Abstractions**: The fundamental execution logic is defined in [`Abstractions.cs`](src/FAI.Core/Abstractions.cs). Refer to this file when explaining how the system works.
- **Python vs C#**: The repository includes Python examples to demonstrate the migration story; when asked about usage, prioritize showing the C# implementation using `PipelineBuilder`.
