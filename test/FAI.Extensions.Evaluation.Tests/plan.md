# Test Plan: FAI.Extensions.Evaluation

This plan outlines the testing strategy for the `FAI.Extensions.Evaluation` project, focusing on the `EvaluationPipeline` and its core abstractions.

## 🎯 Objectives
- Verify the `EvaluationPipeline` correctly orchestrates the data loading, inference, and evaluation flow.
- Ensure `EvaluationPipelineOptions` (batching, parallelism) are respected and function as intended.
- Validate that result aggregation (sample size, runtime) is accurate.
- Test OpenTelemetry integration to ensure activities are correctly created.

## 🏗️ Architecture Under Test
- **Interfaces**: `IDataLoader`, `IInferenceInputGetter`, `IEvaluator`, `IInference` (from `FAI.Core`).
- **Logic**: `EvaluationPipeline`'s state machine for handling `IAsyncEnumerable` streams.
- **Data Models**: `EvaluationPipelineOptions`, `EvaluationPipelineResult`.

## 🧪 Test Scenarios

### 1. Core Integration Flow
- **Scenario**: Full execution from loading to evaluation.
- **Verification**: `IEvaluator` receives the expected output from `IInference`, and `EvaluationPipelineResult` contains the correct `SampleSize`.

### 2. Loading & Batching Logic (`EvaluationPipelineOptions`)
- **No Chunking**: Verify behavior when `LoadingChunkSize` is `null` (collects all data before inference).
- **Chunked Loading**: Verify that data is processed in batches defined by `LoadingChunkSize`.
- **Parallel Loading**: Test when `ParallelLoading` is enabled (uses `Open.ChannelExtensions`).

### 3. Evaluation Parallelism
- **Parallel Evaluation**: Verify that the pipeline correctly handles the switch to parallel evaluation mode via `EvaluationPipelineOptions.ParallelEvaluation`.

### 4. OpenTelemetry Integration
- **Activity Creation**: Verify that `ActivitySource` "fai.evaluation" creates spans for:
    - `fai.evaluation.pipeline`
    - `fai.evaluation.loading`
    - `fai.evaluation.inference`
    - `fai.evaluation.evaluate`
- **Tags**: Check if tags like `fai.evaluation.loaded_count` are correctly populated.

### 5. Error Handling
- Verifying that exceptions in loading or inference propagate correctly and cleanup resources if necessary.

## 🛠️ Implementation Details
- **Framework**: `xunit.v3` with MTP.
- **Mocking**: `NSubstitute` for mocking interfaces.
- **Modern C#**: Use collection expressions `[]` and `System.Threading.Lock` where applicable.
- **Base Directory**: `test/FAI.Extensions.Evaluation.Tests/`

## 📅 Todo List
- [ ] Create mock data types (TestInput, TestOutput).
- [ ] Implement `EvaluationPipelineTests.cs`.
- [ ] Implement `OtelTests.cs` (using `ActivityListener`).
- [ ] Run benchmarks/verification via `dotnet test`.
