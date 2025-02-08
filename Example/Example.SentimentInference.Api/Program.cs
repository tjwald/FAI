using Example.SentimentInference.Model;
using Microsoft.AspNetCore.Mvc;
using ML.Infra;
using ML.Infra.Abstractions;
using Scalar.AspNetCore;

var builder = WebApplication.CreateBuilder(args);
var options = SentimentInferenceOptions.DefaultConfig;

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