# Text embedding example

This end-to-end example runs `sentence-transformers/all-MiniLM-L6-v2` locally with ONNX Runtime and FAI's finite pipeline steps.

The model pipeline performs:

1. Parallel BERT tokenization before batch routing.
2. Token-count ordering to reduce padding.
3. Partitioning by padded-token budget with concurrent ONNX execution.
4. ONNX inference over token IDs, attention masks, and zero-valued token type IDs.
5. Attention-mask mean pooling and L2 normalization directly into one preallocated `[batch, 384]` `Tensor<float>`.

`TextEmbeddingInference.Predict` returns a `[1, 384]` tensor, while `BatchPredict` returns `[batch, 384]`. The pooling stage writes directly into that final contiguous tensor and uses vectorized token-row accumulation without intermediate per-embedding arrays.

The batching defaults match the sentiment example: ascending token-count ordering, a `0.1` maximum padded-token ratio, a `2048` token budget per partition, and up to `10` concurrent partitions. Override these through `TextEmbeddingOptions` for different hardware or workloads.

The console downloads the model and tokenizer from Hugging Face on first run, caches them under `%LOCALAPPDATA%/FAI/models/all-MiniLM-L6-v2`, embeds a small document collection, and ranks it against a query by cosine similarity.

Benchmark mode downloads the 1,500-pair STS Benchmark validation dataset to `%LOCALAPPDATA%/FAI/datasets/stsb`. It reports batched embedding throughput plus Pearson and Spearman correlation against human similarity scores. The dataset is sourced from [`sentence-transformers/stsb`](https://huggingface.co/datasets/sentence-transformers/stsb).

```powershell
dotnet run --project Examples/TextEmbedding/Example.TextEmbedding.Console
dotnet run --project Examples/TextEmbedding/Example.TextEmbedding.Console -- "Which .NET tool works with databases?"
dotnet run -c Release --project Examples/TextEmbedding/Example.TextEmbedding.Console -- --benchmark
```
