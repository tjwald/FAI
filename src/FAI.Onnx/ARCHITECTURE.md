# ONNX Runtime Subsystem Architecture

This directory provides concrete model execution pipelines using Microsoft ONNX Runtime.

## 1. Execution Model

Model execution transforms padded, shaped tensor inputs (`Tensor<long>[]` or `Tensor<float>[]`) into runtime-owned, borrowed tensor output bundles (`TensorOutputs<float>`):

```csharp
public interface IPipeline<Tensor<long>[], TensorOutputs<float>>
```

```mermaid
graph LR
    Input[Tensor&lt;long&gt;[]<br/>Padded Model Inputs] --> Session[InferenceSession / Pool]
    Session --> Ort[ONNX Runtime Native Execution]
    Ort --> Outputs[TensorOutputs&lt;float&gt;<br/>Borrowed OrtValue Bundle]
    Outputs --> Decoder[Synchronous Decoder Callback]
    Decoder --> Dispose[Async / Sync Disposal of OrtValues]
```

### Key Components

- **`OnnxModelExecutorBase`**: Abstract base pipeline wrapping `InferenceSession` and `RunOptions`. Maintains concurrent memory pools for tensor dimension arrays and input buffers to prevent allocation during inference runs.
- **`AsyncOnnxModelExecutor`**: Thread-safe asynchronous executor managing session run calls.
- **`PooledOnnxModelPipeline`**: Implements `IPipeline<Tensor<long>[], TensorOutputs<float>>` on top of an `IObjectPool<OnnxModelExecutorBase>`, allowing concurrent execution across multiple dedicated runtime sessions or devices.
- **`ModelExecutorFactory`**: Instantiates executors based on `ModelExecutorType` (`Simple`, `Pooled`, etc.) and configuration options.

---

## 2. Borrowed Output Lifetime (`TensorOutputs<T>`)

`TensorOutputs<T>` is a Core-owned, disposable bundle of model output tensors.
- Contains live native `OrtValue` instances borrowed from the ONNX Runtime engine.
- Downstream stages (such as `ClassificationDecoding` or multiple-choice decoders) inspect `ReadOnlyTensorSpan<T>` synchronously without copying logits to managed arrays.
- `PipelineOutputDisposer.DisposeAsync` automatically frees native `OrtValue` handles as soon as downstream execution completes, throws, or is cancelled.
- **Zero Logits Materialization**: Unless an application explicitly calls `.ToArray()`, model output logits never allocate on the .NET managed heap.

---

## 3. Fluent Pipeline Builder Integration

`OnnxPipelineBuilderExtensions` exposes `.ThenOnnxModel()` on the unified `PipelineBuilder`:

```csharp
services
    .AddPipeline<ReadOnlyMemory<string>>()
    .Then<ReadOnlyMemory<TokenizedText>, TextTokenization>()
    .UseTokenCountOrdering()
    .UseMaxPaddedTokensPartitioning()
    .Then<Tensor<long>[], TextTensorization>()
    .ThenOnnxModel()
    .Then<Memory<ClassificationResult<bool, float>>, ClassificationDecoding<bool>>()
    .Build();
```

- If `IPipeline<Tensor<long>[], TensorOutputs<float>>` is already registered in `IServiceCollection` (e.g. mock or custom pipeline), `.ThenOnnxModel()` resolves it.
- Otherwise, it resolves `OnnxModelExecutorOptions` and uses `ModelExecutorFactory` to construct the pipeline.

---

## 4. Hardware Agnosticism & Accelerators

Device selection is configured via `OnnxModelExecutorOptions`:
- **CPU**: Default execution provider.
- **GPU Accelerator**: `AppendExecutionProvider_CUDA()` or direct hardware providers when configured in application options.
- Downstream decoders and upstream policies remain completely unaware of whether execution happened on CPU or GPU.
