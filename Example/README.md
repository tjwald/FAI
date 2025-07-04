# Examples

This folder contains example models implemented in python and in C# + FAI.

Each FAI project is split between the following projects:
* *.Model - the inference algorithm and implementation
* *.Console - a batch inference usage example, using the model training data as a benchmark
* Optional - *.Api - an API scenario that shows how to integrate with ASP.NET.

Examples:
* Sentiment Analysis - small NLP classification model. 
  * FAI - [SentimentInference](SentimentInference)
  * python - [python/classification](python/classification)
* Multiple choice - BERT large multiple choice model
  * FAI - [MutlipleChoice](MultipleChoice)
  * python - [python/multiple-choice](python/multiple-choice)
