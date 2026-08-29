using System.Numerics.Tensors;
using Example.TextEmbedding.ConsoleApp;
using Example.TextEmbedding.Model;
using FAI.Core;
using Microsoft.Extensions.DependencyInjection;

string modelDirectory = await MiniLmModelDownloader.EnsureDownloadedAsync();

var services = new ServiceCollection();
var options = TextEmbeddingOptions.Create(modelDirectory) with
{
    UseGpu = !args.Contains("--cpu", StringComparer.OrdinalIgnoreCase)
};
services.AddTextEmbeddingInference(options);
await using ServiceProvider serviceProvider = services.BuildServiceProvider();
var embeddings = serviceProvider.GetRequiredService<TextEmbeddingInference>();

if (args.Contains("--benchmark", StringComparer.OrdinalIgnoreCase))
{
    string datasetPath = await MiniLmModelDownloader.EnsureBenchmarkDatasetDownloadedAsync();
    string performanceDatasetPath = await MiniLmModelDownloader.EnsurePerformanceDatasetDownloadedAsync();
    await StsBenchmark.RunAsync(embeddings, datasetPath, performanceDatasetPath);
    return;
}

string[] documents =
[
    "ASP.NET Core is a cross-platform framework for building web APIs and applications.",
    "Blazor builds interactive web interfaces with C# and HTML.",
    "Entity Framework Core maps .NET objects to relational databases.",
    ".NET MAUI creates native mobile and desktop applications from one codebase.",
    "ML.NET and ONNX Runtime run machine learning models in .NET applications."
];

System.Console.WriteLine("Generating document embeddings with all-MiniLM-L6-v2...");
Tensor<float> documentEmbeddings = await embeddings.BatchPredict(documents);

string query = args.Length == 0
    ? "How can I build a mobile app with .NET?"
    : string.Join(' ', args);
Tensor<float> queryEmbedding = await embeddings.Predict(query);

int dimensions = checked((int)documentEmbeddings.Lengths[1]);
Memory<float> documentBuffer = documentEmbeddings.AsMemory();
Memory<float> queryBuffer = queryEmbedding.AsMemory();

var rankedDocuments = documents
    .Select((document, index) => new
    {
        Document = document,
        Similarity = CosineSimilarity(
            queryBuffer.Span,
            documentBuffer.Span.Slice(index * dimensions, dimensions))
    })
    .OrderByDescending(result => result.Similarity)
    .ToArray();

System.Console.WriteLine($"\nQuery: {query}");
for (int index = 0; index < rankedDocuments.Length; index++)
{
    System.Console.WriteLine($"{index + 1}. {rankedDocuments[index].Similarity:F4}  {rankedDocuments[index].Document}");
}

static float CosineSimilarity(ReadOnlySpan<float> left, ReadOnlySpan<float> right)
{
    if (left.Length != right.Length)
    {
        throw new ArgumentException("Embedding dimensions must match.", nameof(right));
    }

    float dotProduct = 0;
    float leftSquaredNorm = 0;
    float rightSquaredNorm = 0;
    for (int index = 0; index < left.Length; index++)
    {
        dotProduct += left[index] * right[index];
        leftSquaredNorm += left[index] * left[index];
        rightSquaredNorm += right[index] * right[index];
    }

    float denominator = MathF.Sqrt(leftSquaredNorm * rightSquaredNorm);
    return denominator == 0 ? 0 : dotProduct / denominator;
}
