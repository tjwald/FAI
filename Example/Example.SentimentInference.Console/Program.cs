using System.Diagnostics;
using System.Text.Json.Serialization;
using Example.SentimentInference.Model;
using Parquet;
using Parquet.Data;

const string fileName = "train-00000-of-00001.parquet";

var options = SentimentInferenceOptions.DefaultConfig;

var model = await SentimentInferenceFactory.CreateSentimentInference(options);

(string[] input, bool[] expectedOutput) = await LoadTrainingData(fileName);
Console.WriteLine("Finished loading training data");
await RunBatchPredict(model, input, expectedOutput);
return;


static async Task RunBatchPredict(SentimentInference sentimentInference, string[] strings, bool[] bools)
{
    long start = Stopwatch.GetTimestamp();

    bool[] output = await sentimentInference.BatchPredict(strings);
    var end = Stopwatch.GetElapsedTime(start);

    Console.WriteLine($"elapsed time: {end.TotalSeconds}s");
    Console.WriteLine($"avg time: {end.TotalMilliseconds / strings.Length}ms/it");

    int correct = output.Where((t, i) => t == bools[i]).Count();

    Console.WriteLine($"Correct predictions: {correct}/{output.Length}={correct * 1.0 / output.Length}");
}

async Task<(string[] input, bool[] expectedOutput)> LoadTrainingData(string s)
{
    IList<TrainingData> data = await TrainingParquetReader.ReadParquetFileAsync(s);
    string[] strings = data.Select(x => x.Sentence).ToArray();
    Console.WriteLine($"Parquet file loaded with sentences: {strings.Length}");
    bool[] bools = data.Select(x => x.Label == 1).ToArray();
    return (strings, bools);
}

internal class TrainingData
{
    [JsonPropertyName("sentence")] public string Sentence { get; set; } = null!;
    [JsonPropertyName("label")] public long Label { get; set; }
}


internal static class TrainingParquetReader
{
    internal static async Task<List<TrainingData>> ReadParquetFileAsync(string filePath)
    {
        var trainingDataList = new List<TrainingData>();
        await using Stream fs = File.OpenRead(filePath);
        using ParquetReader reader = await ParquetReader.CreateAsync(fs);
        var sentenceField = reader.Schema.FindDataField("sentence");
        var labelField = reader.Schema.FindDataField("label");

        for (int i = 0; i < reader.RowGroupCount; i++)
        {
            using ParquetRowGroupReader rowGroupReader = reader.OpenRowGroupReader(i);
            DataColumn sentenceColumn = await rowGroupReader.ReadColumnAsync(sentenceField);
            DataColumn labelColumn = await rowGroupReader.ReadColumnAsync(labelField);
            for (int j = 0; j < sentenceColumn.Data.Length; j++)
            {
                var trainingData = new TrainingData { Sentence = (string)sentenceColumn.Data.GetValue(j)!, Label = (long)labelColumn.Data.GetValue(j)! };
                trainingDataList.Add(trainingData);
            }
        }

        return trainingDataList;
    }
}