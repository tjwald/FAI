using System.Diagnostics;
using System.Text.Json.Serialization;
using Example.MultipleChoice.Model;
using FAI.Core.Abstractions;
using FAI.Core.ResultTypes;
using FAI.NLP.Tokenization;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Parquet;

const string fileName = "swag_train.parquet";

var builder = Host.CreateApplicationBuilder(args);

var options = SwagMultipleChoiceInferenceOptions.DefaultConfig;

builder.Services.AddDefaultSwagInference(options);

var app = builder.Build();

var model = app.Services.GetRequiredService<IInference<SwagInput, ChoiceResult<TokenizedText>>>();

(SwagInput[] input, int[] expectedOutput) = await LoadTrainingData(fileName);

Console.WriteLine("Finished loading training data");

await RunBatchPredict(model, input, expectedOutput);
return;

static async Task RunBatchPredict(IInference<SwagInput, ChoiceResult<TokenizedText>> sentimentInference, SwagInput[] input, int[] expectedOutput)
{
    long start = Stopwatch.GetTimestamp();

    ChoiceResult<TokenizedText>[] output = await sentimentInference.BatchPredict(input);
    var end = Stopwatch.GetElapsedTime(start);

    Console.WriteLine($"elapsed time: {end.TotalSeconds}s");
    Console.WriteLine($"avg time: {end.TotalMilliseconds / input.Length}ms/it");

    int correct = output.Where((t, i) => t.ChoiceIndex == expectedOutput[i]).Count();

    Console.WriteLine($"Correct predictions: {correct}/{output.Length}={correct * 1.0 / output.Length}");
}

static async Task<(SwagInput[] input, int[] expectedOutput)> LoadTrainingData(string swagTrainParquet)
{
    IList<TrainingData> data = await TrainingParquetReader.ReadParquetFileAsync(swagTrainParquet);
    SwagInput[] strings = data.Select(x => new SwagInput(x.Context, x.Sentence, [x.Ending0, x.Ending1, x.Ending2, x.Ending3])).ToArray();
    int[] labels = data.Select(x => (int)x.Label).ToArray();
    Console.WriteLine($"Parquet file loaded with sentences: {strings.Length}");
    return (strings, labels);
}


file sealed class TrainingData
{
    [JsonPropertyName("sent1")] public string Context { get; set; } = null!;
    [JsonPropertyName("sent2")] public string Sentence { get; set; } = null!;
    [JsonPropertyName("ending0")] public string Ending0 { get; set; } = null!;
    [JsonPropertyName("ending1")] public string Ending1 { get; set; } = null!;
    [JsonPropertyName("ending2")] public string Ending2 { get; set; } = null!;
    [JsonPropertyName("ending3")] public string Ending3 { get; set; } = null!;
    [JsonPropertyName("label")] public long Label { get; set; }
}

file static class TrainingParquetReader
{
    public static async Task<IList<TrainingData>> ReadParquetFileAsync(string filePath)
    {
        var trainingDataList = new List<TrainingData>();
        await using Stream fs = File.OpenRead(filePath);
        using var reader = await ParquetReader.CreateAsync(fs);

        var sent1Field = reader.Schema.FindDataField("sent1");
        var sent2Field = reader.Schema.FindDataField("sent2");
        var ending0Field = reader.Schema.FindDataField("ending0");
        var ending1Field = reader.Schema.FindDataField("ending1");
        var ending2Field = reader.Schema.FindDataField("ending2");
        var ending3Field = reader.Schema.FindDataField("ending3");
        var labelField = reader.Schema.FindDataField("label");

        for (int i = 0; i < reader.RowGroupCount; i++)
        {
            using var rowGroupReader = reader.OpenRowGroupReader(i);

            var sent1Column = await rowGroupReader.ReadColumnAsync(sent1Field);
            var sent2Column = await rowGroupReader.ReadColumnAsync(sent2Field);
            var ending0Column = await rowGroupReader.ReadColumnAsync(ending0Field);
            var ending1Column = await rowGroupReader.ReadColumnAsync(ending1Field);
            var ending2Column = await rowGroupReader.ReadColumnAsync(ending2Field);
            var ending3Column = await rowGroupReader.ReadColumnAsync(ending3Field);
            var labelColumn = await rowGroupReader.ReadColumnAsync(labelField);

            for (int j = 0; j < sent1Column.Data.Length; j++)
            {
                var trainingData = new TrainingData
                {
                    Context = (string)sent1Column.Data.GetValue(j)!,
                    Sentence = (string)sent2Column.Data.GetValue(j)!,
                    Ending0 = (string)ending0Column.Data.GetValue(j)!,
                    Ending1 = (string)ending1Column.Data.GetValue(j)!,
                    Ending2 = (string)ending2Column.Data.GetValue(j)!,
                    Ending3 = (string)ending3Column.Data.GetValue(j)!,
                    Label = (long)labelColumn.Data.GetValue(j)!
                };
                trainingDataList.Add(trainingData);
            }
        }

        return trainingDataList;
    }
}
