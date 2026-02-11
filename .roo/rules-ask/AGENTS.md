# AGENTS.md

This file provides guidance to agents when working in Ask mode within this repository.

## Documentation Rules (Non-Obvious Only)
- **Performance Benchmarks**: Canonical performance gains (7X-14X) are documented in [`README.md`](README.md:20) and compared against standard Python stacks in the [`Examples/`](Examples/) directory.
- **Design Context**: High-level architecture and the motivation for the library (performance on a budget) are found in [`docs/high-level-design.md`](docs/high-level-design.md) and [`docs/Testimonial.md`](docs/Testimonial.md).
- **Core Abstractions**: The fundamental execution logic is defined in [`Abstractions.cs`](src/FAI.Core/Abstractions.cs). Refer to this file when explaining how the system works.
- **Python vs C#**: The repository includes Python examples to demonstrate the migration story; when asked about usage, prioritize showing the C# implementation using `PipelineBuilder`.
