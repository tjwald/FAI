### High-Level Design

```mermaid
graph TD
  Pipeline[Pipeline]
  PipelineBatchExecutor[PipelineBatchExecutor]
  InferenceSteps[InferenceSteps]
  ModelExecutor[ModelExecutor]
  Inference[Inference]

  Inference -- orchestrates --> Pipeline
  Pipeline -- runs via --> PipelineBatchExecutor
  PipelineBatchExecutor -- optimizes execution of --> InferenceSteps
  InferenceSteps -- uses --> ModelExecutor
```
