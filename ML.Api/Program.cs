using Microsoft.AspNetCore.Mvc;
using ML.Infra;
using ML.Infra.Abstractions;
using ML.Infra.ModelExecutors.Onnx;
using ML.Infra.PipelineBatchExecutors;
using ML.Infra.Pipelines;
using ML.Infra.Tokenization;
using Scalar.AspNetCore;
using WebApplication1;

var builder = WebApplication.CreateBuilder(args);

string modelDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "ClassificationModelResources");
var tokenizerOptions = new PretrainedTokenizerOptions(0);

var tokenizer = await TokenizationUtils.BpeTokenizerFromPretrained(modelDir, tokenizerOptions);
var modelExecutor = await OnnxModelExecutor.FromPretrained(modelDir, new OnnxModelExecutorOptions(UseGpu:false));

IPipelineBatchExecutor<string, ClassificationResult<bool>> executor = new SerialPipelineBatchExecutor<string, ClassificationResult<bool>>();

var pipeline = new TextClassificationPipeline<bool>(tokenizer, modelExecutor, new TextClassificationOptions<bool>([false, true]), executor);
var inference = new SentimentInference(pipeline);

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

app.MapPost("/predict", async ([FromBody]string sentence, IInference<string, bool> inference) 
    => await inference.Predict(sentence));

app.MapPost("/predict-orchestrated", async ([FromBody]string sentence, [FromKeyedServices("orchestrated")]IInference<string, bool> inference) 
    => await inference.Predict(sentence));

app.Run();