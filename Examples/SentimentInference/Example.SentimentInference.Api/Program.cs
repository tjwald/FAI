using Example.SentimentInference.Model;
using FAI.Core;
using FAI.Core.Abstractions;
using Microsoft.AspNetCore.Mvc;
using Scalar.AspNetCore;

var builder = WebApplication.CreateBuilder(args);
var options = SentimentInferenceOptions.DefaultConfig;

builder.Services.AddDefaultSentimentInference(options);

// builder.Services.AddKeyedSingleton<IInference<string, bool>>("orchestrated",
//     (sp, _) =>
//     {
//         var inference = sp.GetRequiredService<IInference<string, bool>>();
//         return new InferenceOrchestrator<IInference<string, bool>, string, bool>(
//             new Lazy<IInference<string, bool>>(() => inference), 10, 5, TimeSpan.FromMicroseconds(10)
//             );
//     }
//     );

builder.Services.AddOpenApi();

var app = builder.Build();

if (app.Environment.IsDevelopment())
{
    app.MapOpenApi();
    app.MapScalarApiReference();
}

app.MapPost("/predict", async ([FromBody] string sentence, IInference<string, bool> inference)
    => await inference.Predict(sentence));

// app.MapPost("/predict-orchestrated", async ([FromBody] string sentence, [FromKeyedServices("orchestrated")] IInference<string, bool> inference)
//     => await inference.Predict(sentence));

app.Run();
