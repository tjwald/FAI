---
name: fai-development-workflow
description: "Use when building, testing, linting, formatting, or running console examples in the FAI repository. Covers dotnet CLI commands, pre-commit hooks, testing style with xunit.v3, and running examples in Release mode."
---

# FAI Development Workflow Skill

This skill guides agents when executing builds, tests, code formatting, pre-commit verification, and example console applications in the FAI repository.

---

## 1. Primary Commands

All commands run from the root of the workspace.

| Task | Command | Notes |
| :--- | :--- | :--- |
| **Build** | `dotnet build FAI.slnx` | Fast solution build using modern .slnx format. |
| **Format / Lint** | `dotnet format` | Enforces 4 spaces, `LF` endings, max 160 chars width. |
| **Run All Tests** | `dotnet test` | Runs all 7 test projects using `xunit.v3` and MTP. |
| **Run Single Test Project** | `dotnet test test/FAI.Core.Tests/FAI.Core.Tests.csproj` | Focused test execution for fast iteration. |

### Post-Test Workflow Rules (MANDATORY)
1. **Always run `dotnet format`** after tests pass.
2. **Commit units of work** after tests pass and formatting succeeds.

---

## 2. Running Console Examples

Console examples demonstrate the 7X–14X performance speedup and should always be run with `-c Release`.

### Sentiment Inference Console
Evaluates ~67,349 sentences against SST-2 DistilBERT:
```bash
Push-Location Examples/SentimentInference/Example.SentimentInference.Console
dotnet run -c Release
Pop-Location
```
- **Expected Accuracy**: ~91.87%
- **Expected Throughput**: ~0.05–0.06 ms/it (~3.7s total on GPU).

### Multiple Choice Console
Evaluates ~73,546 SWAG questions against RoBERTa multiple-choice:
```bash
Push-Location Examples/MultipleChoice/Example.MultipleChoice.Console
dotnet run -c Release
Pop-Location
```
- **Expected Accuracy**: ~87.68%
- **Expected Throughput**: ~1.5–1.6 ms/it (~115s total on GPU).

---

## 3. Code Style & Modern C# (.NET 10 / C# 14)

### Collection Expressions
Always use collection expressions `[1, 2, 3]` instead of legacy `new int[] { 1, 2, 3 }` or `new float[] { ... }`:
```csharp
// Correct:
Span<int> indices = [0, 1, 2];
Range[] ranges = [.. partitioner.Partition(input)];

// Incorrect:
int[] indices = new int[] { 0, 1, 2 };
```

### Thread Synchronization
Always use `System.Threading.Lock` instead of `new object()`:
```csharp
// Correct:
private readonly Lock _lock = new();
lock (_lock) { ... }

// Incorrect:
private readonly object _lock = new object();
```

### Formatting & Line Width
- 4 spaces indentation.
- `LF` line endings across all platforms.
- Maximum 160 characters line width.

---

## 4. Testing Conventions (`xunit.v3` & MTP)

- **Library Stability**: When working on test suites, **NEVER** change library code unless implementing a new feature (follow strict TDD).
- **Explicit Assertions**: Use explicit collection matching for ranges and outputs. Avoid partial assertions like `Assert.Single` when full state can be verified.
- **Destination Verification**: When testing decorators or schedulers, verify that destination slices receive exact values and that zero unintended heap allocations occur.
- **Logical Mocks**: Use `LogicalMockModelPipeline` with disposable `TensorOutputs<T>` to verify pipeline orchestration without ONNX engine overhead.
