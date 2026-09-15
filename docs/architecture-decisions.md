# Architecture Decision Records (ADRs)

This document records the key architectural decisions, rationale, and tradeoffs established in FAI.

---

## ADR 1: Removal of `TryAllocateOutput` from Pipeline Contracts

### Context
Earlier designs included `bool TryAllocateOutput(TInput input, out TOutput output)` on `IPreallocatingPipeline`. The goal was allowing upstream callers or decorators to ask a pipeline to synchronously instantiate its output buffer based on input metadata before executing.

### Decision
Remove `TryAllocateOutput` entirely from the pipeline contract hierarchy and rename the interface to `IDestinationPipeline<in TInput, TOutput>`.

### Rationale
1. **Violation of Separation of Concerns**: Pipelines are transformation steps ($T_{\text{in}} \to T_{\text{out}}$), not pre-execution memory allocators.
2. **Failure Across Multi-Stage Pipelines**: In a chain like `Tokenize -> Tensorize -> ONNX -> Decode`, intermediate stages cannot derive the final output buffer shape from raw input without executing the pipeline, which forced manual workaround delegates like `.WithOutputAllocation(...)` in DI.
3. **Redundancy with Batch Traits**: Batch allocation belongs to `IWritableIndexedBatch<TBatch>.AllocateLike(...)`, not individual transformation steps.
4. **Clean Destination Semantics**: Destination execution (`ExecuteAsync(input, destination)`) is about writing into caller-provided or sliced memory without intermediate allocations. It is completely independent of who allocated that memory.

---

## ADR 2: Elimination of `IPipelineChain` and `ExecuteIntoAsync`

### Context
When `TryAllocateOutput` was on `IPreallocatingPipeline`, multi-stage chains could not implement `IPreallocatingPipeline` because they could not preallocate the final output from initial input. Consequently, a parallel interface `IPipelineChain<TInput, TOutput>` was introduced with `ExecuteIntoAsync` and `CanWriteOutput`, along with wrapper classes `PipelineChain`, `PreallocatingPipelineChain`, and `AppendedPipelineChain`.

### Decision
Delete `IPipelineChain<TInput, TOutput>`, `ExecuteIntoAsync`, and the wrapper types. Standardize composition on `AppendedPipeline.Create(prev, next)` implementing `IPipeline<TInput, TOutput>` and `IDestinationPipeline<TInput, TOutput>`.

### Rationale
Once `TryAllocateOutput` was removed, `ExecuteIntoAsync` had the exact same signature, semantics, and behavior as `IDestinationPipeline.ExecuteAsync(input, destination)`. Maintaining two parallel interface hierarchies created duplicate abstractions, unnecessary runtime type checks, and awkward double-casting in the fluent builders.

---

## ADR 3: Rejection of Unified `Fanout` (Partitioning vs. Routing)

### Context
We evaluated unifying `PartitioningPipeline` and `RoutingPipeline` into a single `FanoutPipeline` abstraction, hypothesizing that both divide a batch, execute through an `IPartitionScheduler`, and reassemble results.

### Decision
Keep `PartitioningPipeline` and `RoutingPipeline` as distinct, first-class batch policies.

### Rationale
The fundamental difference is the underlying memory access pattern:
- **`PartitioningPipeline` (Data Parallelism)**: Operates on contiguous subranges (`Range`). Input and destination are sliced in $O(1)$ without copying memory (`_inputBatch.Slice`). Execution is naturally order-preserving and zero-allocation.
- **`RoutingPipeline` (Model / MoE Routing)**: Operates on arbitrary, non-contiguous index subsets (`int[]`). Inputs must be gathered into rented memory (`_inputBatch.Gather`), targets execute heterogeneous models, and outputs must be permuted back into original positions.

Unifying them would either force partitioning to use non-contiguous gathering (destroying its zero-allocation property) or leak routing metadata into the pipeline type signature.

---

## ADR 4: Sort-by-Route in `RoutingPipeline`

### Context
Naive routing evaluates $K$ routes by performing $K$ separate gathers, executing return-value pipelines, allocating $K$ outputs, and scattering each result.

### Decision
Implement the Sort-by-Route pattern inside `RoutingPipeline`:
1. Partition the destination buffer into contiguous slices matching route cardinality.
2. Targets execute into their assigned contiguous slices (`_outputBatch.Slice(destination, range)`).
3. Once all routes complete, a single in-place cycle permutation (`_outputBatch.PermuteInPlace`) restores original batch order.

### Rationale
This minimizes memory allocations and reduces scatter overhead. When targets implement `IDestinationPipeline`, execution writes directly into contiguous slices of the final destination buffer with zero intermediate aggregate allocations.

---

## ADR 5: Elimination of `WritableOperations` Reflection

### Context
`IndexedBatchOperations.GetWritable<TBatch>()` used runtime reflection (`MakeGenericType` and `Activator.CreateInstance`) to inspect `TBatch` and instantiate batch traits.

### Decision
Remove static reflection and resolve batch operations through Microsoft Dependency Injection:
- Expose typed registration methods: `services.AddMemoryBatch<T>()`, `services.AddTensorBatch<T>()`, `services.AddBatchOperations<TBatch, TOps>()`.
- Decorators resolve batch traits directly from `IServiceProvider`: `serviceProvider.GetRequiredWritableBatch<TOutput>()`.

### Rationale
- **AOT & Trimming**: Dynamic generic construction at runtime violates Native AOT and trimming requirements in .NET 9/10.
- **Explicit DI**: In a DI-driven framework, batch operations are services that should be configurable, mockable, and discoverable in the service collection.
- **Actionable Errors**: If a batch trait is missing, DI throws an explicit exception indicating the exact missing type and how to register it.

---

## ADR 6: Consolidation of Pipeline Builders into `PipelineBuilder<TStart, TCurrent>`

### Context
The fluent API was previously split into three distinct types:
- `PipelineBuilder<TInput>` (arity 1)
- `ComposedPipelineBuilder<TStart, TCurrent>` (arity 2)
- `DecoratedPipelineBuilder<TStart, TBoundary, TCurrent>` (arity 3)

This caused an arity explosion where every method (`Then`, `Fork`, `Use`, `Build`) and every extension method (`ThenOnnxModel`, `UseTokenCountOrdering`, etc.) had to be duplicated across all three classes.

### Decision
Unify into a single public builder `PipelineBuilder<TStart, TCurrent>` that encapsulates an internal state machine `IStage<TStart, TCurrent>` (`InitialStage`, `ComposedStage`, `DecoratedStage`).

### Rationale
- Callers only care about the pipeline's start type `TStart` and current type `TCurrent`. The internal decoration boundary `TBoundary` was a leaked implementation detail.
- Eliminates 2 redundant classes and hundreds of lines of duplicated code.
- Reduces extension methods (e.g. `ThenOnnxModel`) from 3 overloads to a single generic extension.

---

## ADR 7: Pre-Wrapping Routing Targets as `IDestinationPipeline`

### Context
`RoutingPipeline` previously handled targets typed as `IPipeline<TInput, TOutput>`, requiring a static fallback method `DestinationPipeline.ExecuteAsync(...)` that inspected targets at runtime during every batch.

### Decision
Type `BatchRoute.Target` strictly as `IDestinationPipeline<TInput, TOutput>`. Routing strategies pre-wrap return-only targets once during strategy construction using `target.AsDestinationPipeline(outputBatch)`.

### Rationale
- Eliminates runtime type testing (`target is IDestinationPipeline`) and conditional branching inside the hot per-batch routing loop.
- Brings `RoutingPipeline` into architectural alignment with `OrderingPipeline` and `PartitioningPipeline`, which both normalize inner targets during construction.
