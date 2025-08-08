using Example.SentimentInference.Console;
using Example.SentimentInference.Model;
using FAI.Core.Abstractions;

const string fileName = "train-00000-of-00001.parquet";

var options = SentimentInferenceOptions.DefaultConfig;

IInference<string, bool> model = await SentimentInferenceFactory.CreateSentimentInference(options);

var evaluationManager = new EvaluationManager(fileName);
await evaluationManager.LoadData();
await evaluationManager.Run(model);
return;