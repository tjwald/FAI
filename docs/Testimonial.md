## What Got Us Here

In a company @tjwald worked for, there were many custom small nlp models for different tasks, mostly classification and
multiple choice tasks.

These were wrapped in complex algorithms requiring multiple queries to different models or the same model multiple times
for the same incoming request.

The models were hosted on a single thread/process FastAPI with no async implementation - meaning only processing one
query at a time.

One of the models was processing roughly 200 requests a minute in spike traffic, but was receiving 20K requests per
minute. That means we needed 1000 GPU machines to handle the load, but we limited it to 100 and relied on retries to
deal with the timeout requests.

The machines were also only using ~1% of the GPU so something had to be done.

## Optimizations Work

After many optimizations and a month of work, optimizing our FastAPI server and migrating to ONNX for the model
inference, I got it to 28K requests a minute, and down to 1 server.

What were some of the bottlenecks I solved?

* Efficient use of async await - processing CPU and GPU computation concurrently
* Dynamic Batching - GPUs love batches, but we were getting single sentence requests from different processes. So we
  collect them and run them together.
* Using Onnx - pytorch is a training framework, both heavy in installations (Fatter Image), but also slower in runtime.
  Migrating to ONNX solved both problems - reduced our image size by 60%, and our inference time from X4 to X20
  depending on the algorithm.
* Algorithmic Improvements - complex algorithms, with many ping pongs between the gpu and cpu, do not easily optimize.

But we were reaching the limitations of python - our GPU was less than 30% utilized, and our CPU was spinning with
increasing latency with increased load.

## .Net 9.0

After all of this work, dotnet 9 came out with many improvements to AI workloads, including the missing piece from our
point of view - tokenizers!
As all of our models were NLP based, we needed tokenizers to be able to implement our models in C#, and now we had them.
In addition the new `Tensor<T>` type really helps!

In 2 days work on my own, I was able to create the initial foundation of this library using ASP.NET framework with ONNX
for the model runtime, and benchmark a similar model.
and guess what?

I got 200K+ requests a minute.

I asked the company if this was something they wanted to invest in, but as they didn't need this extra throughput, and
adding a new tech-stack to a python-only shop didn't make sense they did not invest in this library.

## What did we learn?

Python was just fine for the company, and most impactful optimizations are in the framework level and algorithmic
level (from 200reqs/m -> 24K reqs/m remember?)

Before you abandon python for C# for the performance gain, remember that C# is not as mature as python in the AI
ecosystem.

Most new innovations happen in python and would require porting to C#.

### Notes:

* This simulates training, and batched inference - However, my issue was serving dynamic queries in spike loads.
* You can definitely take many of these optimizations and apply them to the python solution.
* In a dynamic server setting, you will find that these benchmarks scale very well in C# but not in python. See
  `InferenceOrchestrator<TInference, TQuery, TResult>`
* The more CPU logic in the inference algorithm surrounding the underlying model, the better these benchmarks favor C#
  even with all the optimizations applied to python.
* There are many Gen 0 allocations in the current solution. This is due to how `Tensor<T>` is implemented, and is under
  discussion
  in [PyTorch & HuggingFace Custom Models Migration Story](https://github.com/microsoft/semantic-kernel/issues/9793)  
