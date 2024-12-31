This project is designed to:
* Showcase the new features of dotnet 9.0 in the world of AI
* Compare the performance of C# to python
* Show cool AI optimizations techniques for runtime performance

This project also has many points that shows we could improve the framework and language, and to support the ML Ops story of migrating from python to a more performant solution.


## Background 

Many AI projects start from a researcher developing or customizing existing models to specific tasks.

These are usually developed in python, normally using the [HuggingFace transformers library](https://huggingface.co/docs/transformers/en/index), and pytorch models.

However, getting these models to run in production in an efficient manner is a different story.

In my case, we had multiple custom nlp models with complex algorithms deciding how and on what to use them.

These models were bleeding a third of our budget on in production.

For example, one of the models was processing 200 requests a minute. We needed it to process 20K requests a minute -> we need 1000 machines (we limited it to 100...).

After many optimizations and a month of work, optimizing our FastAPI server and migrating to ONNX for the model inference, we got it to 28K requests a minute, and down to 1 server.

What were some of the bottlenecks we solved?
* Efficient use of async await - processing cpu and gpu workloads at the same time
* Dynamic Batching - gpus love batches, but we were getting single sentence requests from different processes
* Using Onnx - pytorch is a training framework, both heavy in installations, but also slower in runtime.
* Algorithmic Improvements - complex algorithms, with many ping pongs between the gpu and cpu, do not easily optimize.


### .Net 9.0

After all of this work, dotnet 9 came out with many improvements to AI workloads, including the missing piece - tokenizers!

In 2 days work, I was able to get most of my optimizations over to the ASP.NET framework, with ONNX for the model runtime, 
and guess what?

We got 200K+ requests a minute.

The company didn't need this, and adding a new tech-stack to a python-only shop didn't make sense. 



## The Project

The library [ML.Infra](ML.Infra) provides the foundation for optimizing your own models.

The repo also contains an example C# usage in the `Example.SemanticInference.*` folders.

This project demonstrates how to optimize the use of AI models, and get better performance by migrating to C#.

This doesn't mean this is all good - there are many issues documented in: [PyTorch & HuggingFace Custom Models Migration Story](https://github.com/microsoft/semantic-kernel/issues/9793)

Also, since the model I use as an example in this project has a custom tokenizer, I did my best to use the most similar tokenizer available in C#.

Under [example/python](/example/python) you will find the hugging face model I am using as a baseline. This model was picked at random, as it is a small nlp model, that is well up-voted on HuggingFace.
This model is smaller than the one mentioned previously

In the Example.SemanticInference.Model you will find the optimized onnx model with the needed resources.


## The Lego Bricks

These can be mixed and matched to tailor the performance and behaviour of most NLP models.

### Model Executors
These actually run the model. You can switch from ONNX to another implementation without effecting the other building blocks.

You can also see that there are multiple ONNX runners, with a pooled wrapper that can pool multiple instances. 

### Tokenizers
I didn't do much except implement a wrapper to the dotnet abstraction of a tokenizer such that it supports batching and returns `Tensor<T>`.


### Pipelines
I was inspired by [HuggingFace](https://huggingface.co/docs/transformers/en/index), but I added my own little twist - you can now inject a custom `IPipelineBatchExecutor<TInput, TOutput>` that controls how batches are executed.

There are multiple examples of these PipelineBatchExecutors:
* Serial - just runs them in a loop one after the other.
* Parallel - will do just that - parallelize the batches.
* OutOfOrder - this one is tricky, but since different sentences translate to different sizes, and the GPU likes the same size, batching similar sized sentences sometimes helps with performance. However, we expect the ordering of the batch to stay the same, so we have to sort twice - once by length, and once by original index. 


## Results:

### My Specs: 
* CPU: i5-13600KF
* GPU: RTX 3060 12GB
* CUDA (C#) - 12.6
* CUDA (Torch) - 12.1
* Torch - 2.5.1
* Onnx - 1.20.1

### Data Set:
I am using the training data set, in the parquet provided in the folder.

67349 Sentences with the relevant labels

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
* Threading: 50
* Time: 10s
* Time per sentence: 0.16 ms


A little over 5X improvement on a small model with no logic outside the model.

### Notes:
* This simulates training, and batched inference - However, my issue was serving dynamic queries in spike loads.
* You can definitely take many of these optimizations and apply them to the python solution.
* In a dynamic server setting, you will find that these benchmarks scale very well in C# but not in python. See `InferenceOrchestrator<TInference, TQuery, TResult>`
* The more CPU logic in the inference algorithm surrounding the underlying model, the better these benchmarks favour C# even with all the optimizations applied to python.
* There are many Gen 0 allocations in the current solution. This is due to how `Tensor<T>` is implemented, and is under discussion in [PyTorch & HuggingFace Custom Models Migration Story](https://github.com/microsoft/semantic-kernel/issues/9793)  


### What should you do?
Probably nothing. I hope you will learn something from this project.

Python was just fine for the company, and most impactful optimizations are in the framework level and algorithmic level (from 200reqs/m -> 24K reqs/m remember?)

Before you abandon python for C# for the performance gain, remember that C# is not as mature as python in the AI ecosystem.

Most new innovations happen in python and would require porting to C#.  