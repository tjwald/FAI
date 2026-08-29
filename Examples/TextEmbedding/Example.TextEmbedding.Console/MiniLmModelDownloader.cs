namespace Example.TextEmbedding.ConsoleApp;

internal static class MiniLmModelDownloader
{
    private const string RepositoryUrl = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main";
    private const string BenchmarkDatasetUrl = "https://huggingface.co/datasets/sentence-transformers/stsb/resolve/main/data/validation-00000-of-00001.parquet";
    private const string PerformanceDatasetUrl = "https://huggingface.co/datasets/fancyzhx/ag_news/resolve/main/data/train-00000-of-00001.parquet";
    private static readonly (string RemotePath, string LocalName)[] Files =
    [
        ("onnx/model.onnx", "model.onnx"),
        ("vocab.txt", "vocab.txt"),
        ("tokenizer_config.json", "tokenizer_config.json")
    ];

    public static async Task<string> EnsureDownloadedAsync(CancellationToken cancellationToken = default)
    {
        string modelDirectory = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "FAI",
            "models",
            "all-MiniLM-L6-v2");
        Directory.CreateDirectory(modelDirectory);

        using var httpClient = new HttpClient
        {
            Timeout = TimeSpan.FromMinutes(10)
        };

        foreach ((string remotePath, string localName) in Files)
        {
            string destination = Path.Combine(modelDirectory, localName);
            await DownloadIfMissingAsync(httpClient, $"{RepositoryUrl}/{remotePath}", destination, cancellationToken);
        }

        return modelDirectory;
    }

    public static async Task<string> EnsureBenchmarkDatasetDownloadedAsync(CancellationToken cancellationToken = default)
    {
        string datasetDirectory = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "FAI",
            "datasets",
            "stsb");
        Directory.CreateDirectory(datasetDirectory);

        string datasetPath = Path.Combine(datasetDirectory, "validation.parquet");
        using var httpClient = new HttpClient
        {
            Timeout = TimeSpan.FromMinutes(10)
        };
        await DownloadIfMissingAsync(httpClient, BenchmarkDatasetUrl, datasetPath, cancellationToken);
        return datasetPath;
    }

    public static async Task<string> EnsurePerformanceDatasetDownloadedAsync(CancellationToken cancellationToken = default)
    {
        string datasetDirectory = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "FAI",
            "datasets",
            "ag-news");
        Directory.CreateDirectory(datasetDirectory);

        string datasetPath = Path.Combine(datasetDirectory, "train.parquet");
        using var httpClient = new HttpClient
        {
            Timeout = TimeSpan.FromMinutes(10)
        };
        await DownloadIfMissingAsync(httpClient, PerformanceDatasetUrl, datasetPath, cancellationToken);
        return datasetPath;
    }

    private static async Task DownloadIfMissingAsync(
        HttpClient httpClient,
        string sourceUrl,
        string destination,
        CancellationToken cancellationToken)
    {
        if (File.Exists(destination) && new FileInfo(destination).Length > 0)
        {
            return;
        }

        System.Console.WriteLine($"Downloading {Path.GetFileName(destination)}...");
        string temporaryPath = destination + ".download";
        try
        {
            using HttpResponseMessage response = await httpClient.GetAsync(
                sourceUrl,
                HttpCompletionOption.ResponseHeadersRead,
                cancellationToken);
            response.EnsureSuccessStatusCode();

            await using (Stream source = await response.Content.ReadAsStreamAsync(cancellationToken))
            await using (FileStream destinationStream = File.Create(temporaryPath))
            {
                await source.CopyToAsync(destinationStream, cancellationToken);
            }

            File.Move(temporaryPath, destination, true);
        }
        finally
        {
            File.Delete(temporaryPath);
        }
    }
}
