## Sentiment Analysis Example

This folder showcases how to use this library for optimizing static label predictions - classification task.

The model I chose is a small BERT model from [HuggingFace](https://huggingface.co/distilbert/distilbert-base-uncased) since it is a small nlp and highly upvoted.

Since the model uses a custom tokenizer, there is a certain loss of accuracy between python and C# implementations, as well as a difference in quantization (discussed later).

The python implementation for running this model on the same data can be found under [example/python/classification](../../Example/python/classification).

In the Example.SemanticInference.Model project folder you will find the optimized onnx model with the needed resources, while in the python folder you will find the original pytorch model.


## Setup

### My Specs:
* CPU: i5-13600KF
* GPU: RTX 3060 12GB
* CUDA (C#) - 12.6
* CUDA (Torch) - 12.1
* Torch - 2.5.1
* Onnx - 1.20.1

### Data Set:
I am using the training data set, in the parquet provided in the folder. The idea is to test speed, so there is no issue with reusing the training set.

67349 Sentences with the relevant labels


## Results:

### Python:
* Torch + GPU
* Batch Size: 20
* Threading: 1
* Time: 51s
* Time per sentence: 0.77 ms

### C#:
* ONNX + GPU
* ORT_SEQUENTIAL
* Batch Size: 12
* Threading: 4
* Time: 10.56s
* Time per sentence: 0.157 ms


A 5X improvement on a small model with no logic outside the model.

After further optimizations and fixing token-count ordering I was able to further reduce the time taken.

### C#
* ONNX + GPU
* ORT_SEQUENTIAL
* Batch Size: 12
* Threading: 4
* Time: 6.04s
* Time per sentence: 0.089 ms

8.65X faster than python.  This enabled increasing the batch size:

### C#
* ONNX + GPU
* ORT_SEQUENTIAL
* **Batch Size: 400**
* Threading: 4
* Time: 3.72s
* Time per sentence: 0.055 ms

13.63X faster than python.

The time per batch size (4 threads):

| Batch Size | Time (s)    |
|------------|-------------|
| 12         | 5.86        |
| 20         | 5.05        |
| 50         | 4.19        |
| 100        | 3.94        |
| 200        | 3.87        |
| 300        | 3.79        |
| 400        | 3.75  <---- |
| 450        | 3.78        |
| 500        | 3.86        |

With max-padded-token partitioning and token-count ordering we get:

* ONNX + GPU
* ORT_SEQUENTIAL
* **Token Max Batch Size: 2048**
* Threading: 4
* Time: 3.64s
* Time per sentence: 0.054 ms

When checking using a profiler, the time spent in the libraries C# code is down to 0.1%, and GPU utilization is at 98% during this run.

### Notes:
* This simulates training, and batched inference - However, my issue was serving dynamic queries in spike loads.
* You can definitely take many of these optimizations and apply them to the python solution.
* In a dynamic server setting, you will find that these benchmarks scale very well in C# but not in python. See `InferenceOrchestrator<TInference, TQuery, TResult>`
* The more CPU logic in the inference algorithm surrounding the underlying model, the better these benchmarks favor C# even with all the optimizations applied to python.
* There are many Gen 0 allocations in the current solution. This is due to how `Tensor<T>` is implemented, and is under discussion in [PyTorch & HuggingFace Custom Models Migration Story](https://github.com/microsoft/semantic-kernel/issues/9793)
