# Test Plan: FAI.Onnx

This plan outlines the testing strategy for the `FAI.Onnx` project, focusing on unit testing utility classes, pool logic, and the model executor infrastructure.

## 1. Goals
- Ensure `CircularAtomicCounter` is thread-safe and wraps correctly.
- Verify `OnnxTensorUtils` correctly converts memory spans to `OrtValue` arrays.
- Validate `MultiDeviceObjectPool` and `OnnxModelExecutorObjectPool` round-robin logic.
- Verify `ModelExecutorFactory` creates the correct executor types based on configuration.
- Test `OnnxModelExecutor` and `AsyncOnnxModelExecutor` using a minimal ONNX model to verify the execution flow and async behavior.

## 2. Testing Strategy

### 2.1 Unit Tests (Logic Only)
- **`CircularAtomicCounter`**:
    - Test wrap-around behavior (e.g., if max is 3, sequence should be 0, 1, 2, 0...).
    - Test thread safety by incrementing from multiple threads and verifying the final state/distribution.
- **`OnnxTensorUtils`**:
    - Verify `ToOrtValues` creates the expected number of `OrtValue` objects with correct dimensions.

### 2.2 Component Tests (Architecture & Pools)
- **`MultiDeviceObjectPool`**:
    - verify it returns executors in the expected round-robin order.
- **`ModelExecutorFactory`**:
    - Test that `CreateModelExecutor` returns:
        - `PooledModelExecutor` when `MultiDeviceExecutorOptions` or `PooledExecutorOptions` are provided.
        - `OnnxModelExecutor` (or variants) for simple configurations.

### 2.3 Integration Tests (Minimal ONNX)
- **Minimal Model**: Create or use a tiny ONNX model (e.g., an identity function or simple addition) that remains in memory or is loaded from a small byte array to avoid CI overhead.
- **`OnnxModelExecutor`**:
    - Verify synchronous-like execution through the `Run` method.
- **`AsyncOnnxModelExecutor`**:
    - Verify it correctly handles async execution using `Session.RunAsync`.
    - Validate its specific behavior regarding output memory allocation.

## 3. Implementation Details
- **Framework**: `xunit.v3` with Microsoft Testing Platform (MTP).
- **Mocking**: `NSubstitute` for mocking dependencies where applicable (though `InferenceSession` may require a real instance or a byte-array based initialization).
- **Modern C#**: Utilize collection expressions `[]`, primary constructors (where applicable), and `System.Threading.Lock`.

## 4. Execution Workflow
1. Create `test/FAI.Onnx.Tests/plan.md` (this file).
2. Implement unit tests for `Utils` and `Pools`.
3. Implement `ModelExecutorFactory` tests.
4. Implement executor tests using a minimal model.
5. Run tests using `dotnet test`.
6. Apply `dotnet format`.
7. Commit changes.
