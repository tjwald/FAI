# Package Diagram

```mermaid
graph TD
  Core[FAI.Core\nstep contracts, leases, policies]
  DI[FAI.Core.Extensions.DI\ntyped step builder]
  NLP[FAI.NLP\ntokenization and text task steps]
  NLPDI[FAI.NLP.Extensions.DI\ntext policy decorators]
  Vision[FAI.Vision\nimage task steps]
  ONNX[FAI.Onnx\nmodel steps and runtime pools]
  App[Application inference facade]

  Core --> DI
  Core --> NLP
  Core --> Vision
  Core --> ONNX
  DI --> NLPDI
  NLP --> NLPDI
  DI --> App
  NLPDI --> App
  ONNX --> App
```

## Extension Model

Runtime packages implement `IStep` for model-specific disposable tensor outputs. Domain packages implement task steps and policies without introducing another execution abstraction. Steps that can derive storage synchronously from metadata may also implement `IPreallocatingStep`. Applications compose those pieces with `AddPipeline<TInput>()` and `Then<TOutput, TStep>()`, then expose an `IInference<TInput, TOutput>` facade where needed.
