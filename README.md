# FAI - Fast AI on A Budget

[![Build Status](https://github.com/tjwald/FAI/actions/workflows/publish-nuget.yml/badge.svg)](https://github.com/tjwald/FAI/actions/workflows/publish-nuget.yml)
[![NuGet](https://img.shields.io/nuget/v/FAI.Core.svg)](https://www.nuget.org/packages/FAI.Core)
[![License](https://img.shields.io/github/license/tjwald/FAI)](https://github.com/tjwald/FAI/blob/main/LICENSE)

<img src="/docs/Images/FAI%20Logo.jpg" alt="FAI logo" width="300" height="300"/></br>

FAI (Fast AI) is a library design to maximize the performance of your models by giving you the best tools to do so:

* Choose the right hardware for you
* Choose the right model executing framework for you
* Implement an inference task once and inject the right hardware and framework to run it for the specific model
* Optimize batch inference execution with custom algorithms and scheduling
* Enjoy a nice User Friendly API for consuming code via Pipeline abstraction
* Bootstrap inference algorithms for different applications by providing a common abstraction to build on.

[![AI Community Standup Video](https://img.youtube.com/vi/ptdNWGj8CN8/0.jpg)](https://www.youtube.com/watch?v=ptdNWGj8CN8)

## 📦 NuGet Packages

| Package | Version | Description |
|---------|---------|-------------|
| [FAI.Core](https://www.nuget.org/packages/FAI.Core/) | ![NuGet](https://img.shields.io/nuget/v/FAI.Core) | Core abstractions for scheduling & batching |
| [FAI.Core.Extensions.DI](https://www.nuget.org/packages/FAI.Core.Extensions.DI/) | ![NuGet](https://img.shields.io/nuget/v/FAI.Core.Extensions.DI) | Dependency injection extensions for FAI.Core |
| [FAI.NLP](https://www.nuget.org/packages/FAI.NLP/) | ![NuGet](https://img.shields.io/nuget/v/FAI.NLP) | NLP tasks and inference |
| [FAI.NLP.Extensions.DI](https://www.nuget.org/packages/FAI.NLP.Extensions.DI/) | ![NuGet](https://img.shields.io/nuget/v/FAI.NLP.Extensions.DI) | Dependency injection extensions for FAI.NLP |
| [FAI.Onnx](https://www.nuget.org/packages/FAI.Onnx/) | ![NuGet](https://img.shields.io/nuget/v/FAI.Onnx) | OnnxRuntime Integration |

## Results

Under the [Examples](Examples) folder, you can find projects written using the standard python stack and compare it with
the same task written using this library. </br>
The results in the examples are anywhere between 7X and 14X that is for offline batch inference, for online Web
inference, the gains can be a lot more.

## Background

Many AI projects start from a researcher developing or customizing existing models to specific tasks.

These are usually developed in python, using
the [HuggingFace transformers library](https://huggingface.co/docs/transformers/en/index), and pytorch models.

However, getting these models to run in production in an efficient manner is a different story.

See the [Testimonial](docs/Testimonial.md) for how this library got to be created.

### .Net 9.0

Dotnet 9 came out with many improvements to AI workloads including:

- `Tensor<T>`
- `Tokenizer`

The missing pieces for a generic ML library designed for performance.

---

## This Project

FAI is designed to:

* Support the ML-Fin-Ops migration story from python to production
* Enable more AI for less within a budget
* Bootstrap common usage patterns of ML with high performance

The repo also contains an example C# usage in the `Example` folder.

This project demonstrates how to optimize the use of AI models, and get better performance by migrating to C#.

This doesn't mean this is all good - there are many issues documented
in: [.NET AI Fundamentals Feedback / Feature Requests](https://github.com/dotnet/machinelearning/issues/7383)

We would love contributions -

* More ML tasks - QA, Entity recognition, etc.
* More Input modalities - Image, Video, Multi-Modals etc.
* More Model Inference Frameworks - PyTorch via TorchSharp/CSnakes, etc.
* More Bootstrapping - ASP.NET Autowired web server, Evaluation pipeline, configuration and dependency injection
  integration, etc.

## General Note

Python is and probably will be the foundation for AI/ML research and development for the coming years.

This practically means that any cutting edge new ML result will take time to find its way to C#+dotnet,
and you **should** take this into consideration.

If the models you are developing are not under a lot of dynamic load, you aren't using your entire budget on running AI
with low utilization, then maybe this is not worth the effort to migrate.

---

## The Lego Bricks

These can be mixed and matched to tailor the performance and behaviour of most NLP models.

### Model Executors
These actually run the model. You can switch from ONNX to another implementation without effecting the other building blocks.

You can also see that there are multiple ONNX runners, with a pooled wrapper that can pool multiple instances.

Currently supported:

* FAI.Onnx - implements multiple onnx ModelExecutors. When referencing this package, you need to add the specific Onnx
  package you want to use (GPU, OpenVino etc.).

### Tokenizers

A batch tokenizer implementation - `FAI.NLP.Tokenization.PretrainedTokenizer`

### Pipelines
I was inspired by [HuggingFace](https://huggingface.co/docs/transformers/en/index), but I added my own little twist - you can now inject a custom `IPipelineBatchExecutor<TInput, TOutput>` that controls how batches are executed.

### Pipeline Batch Executors

This abstraction enables you to mix and match different batching and scheduling algorithms to get the most performance
out of you model and hardware.

There are multiple examples of these PipelineBatchExecutors:
* Serial - just runs them in a loop one after the other.
* Parallel - will do just that - parallelize the batches.
* TokenCountSorting - this one is tricky, but since different sentences translate to different sizes, and the GPU likes the same size, batching similar sized sentences sometimes helps with performance. However, the user expects the ordering of the batch to stay the same, so we have to sort twice - once by length, and once by original index.
* TokenBatchSize - group the sentences up to a token threshold instead of sentence count.
* Max Padding - make sure you don't waste a lot of compute on padding tokens.
  And more.

---

## Feedback and Contributions

We would love your feedback! Please file issues on this repo with suggestions and improvements you would like to
contribute.

Please file an issue **before** you open a PR so we get a chance to review the suggestion and make sure the design and
implementation direction are in alignment with this project.
