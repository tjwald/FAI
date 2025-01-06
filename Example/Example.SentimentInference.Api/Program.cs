using Microsoft.AspNetCore.Mvc;
using Microsoft.ML.OnnxRuntime;
using ML.Infra;
using ML.Infra.Abstractions;
using ML.Infra.ModelExecutors.Onnx;
using ML.Infra.Tokenization;
using Example.SentimentInference.Model;
using Scalar.AspNetCore;

var builder = WebApplication.CreateBuilder(args);
SentimentInferenceOptions options = new SentimentInferenceOptions(
    ModelDir: Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "ClassificationModelResources"),
    TokenizerOptions: new PretrainedTokenizerOptions(PaddingToken: 0),
    new OnnxModelExecutorOptions(UseGpu: true, ExecutionMode: ExecutionMode.ORT_SEQUENTIAL, MaxInferenceSessions: 1, MaxThreads: null),
    MaxConcurrency: 50,
    BatchSize: 12,
    UseOutOfOrderExecution: false,
    ModelExecutorType: ModelExecutorType.Simple
);
var inference = await SentimentInferenceFactory.CreateSentimentInference(options);

builder.Services.AddSingleton<IInference<string, bool>>(inference);
builder.Services.AddKeyedSingleton<IInference<string, bool>>("orchestrated",
    new InferenceOrchestrator<SentimentInference, string, bool>(new Lazy<SentimentInference>(() => inference), 10, 5, TimeSpan.FromMicroseconds(10)));

builder.Services.AddOpenApi();

var app = builder.Build();

if (app.Environment.IsDevelopment())
{
    app.MapOpenApi();
    app.MapScalarApiReference();
}

app.MapPost("/predict", async ([FromBody] string sentence, IInference<string, bool> inference)
    => await inference.Predict(sentence));

app.MapPost("/predict-orchestrated", async ([FromBody] string sentence, [FromKeyedServices("orchestrated")] IInference<string, bool> inference)
    => await inference.Predict(sentence));

app.Run();