# Package Diagram

```mermaid
graph TD
  Core[FAI.Core\npipeline contracts, leases, policies]
  DI[FAI.Core.Extensions.DI\ntyped pipeline builder]
  NLP[FAI.NLP\ntokenization and text task pipelines]
  NLPDI[FAI.NLP.Extensions.DI\ntext policy decorators]
  Vision[FAI.Vision\nimage task pipelines]
  ONNX[FAI.Onnx\nmodel pipelines and runtime pools]
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

Runtime packages implement `IPipeline` for model-specific disposable tensor outputs. Domain packages implement task pipelines and policies without introducing another execution abstraction. Pipelines that can derive storage synchronously from metadata may also implement `IPreallocatingPipeline`. Applications compose those pieces with `AddPipeline<TInput>()` and `Then<TOutput, TPipeline>()`, then expose an `IInference<TInput, TOutput>` facade where needed.
