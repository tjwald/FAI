using System.Diagnostics;
using System.Text.Json.Serialization;
using FAI.Core.Abstractions;
using Parquet;
using Parquet.Data;
using SysConsole = System.Console;


namespace Example.SentimentInference.Console;

public class EvaluationManager
{
    private readonly string _fileName;
    private string[]? _inputs;
    private bool[]? _expectedOutputs;

    public EvaluationManager(string fileName)
    {
        _fileName = fileName;
    }

    public async Task LoadData()
    {
        (_inputs, _expectedOutputs) = await LoadTrainingData(_fileName);
        SysConsole.WriteLine("Finished loading training data");
    }

    public async Task Run(IInference<string, bool> model, int? count = null)
    {
        if (_inputs == null || _expectedOutputs == null)
        {
            throw new InvalidOperationException("Must call LoadData before calling this method");
        }

        Memory<string> input = _inputs.AsMemory();
        Memory<bool> expectedOutput = _expectedOutputs.AsMemory();
        if (count != null)
        {
            input = input[..count.Value];
            expectedOutput = expectedOutput[..count.Value];
        }

        await RunBatchPredict(model, input, expectedOutput);
    }

    static async Task RunBatchPredict(IInference<string, bool> sentimentInference, Memory<string> strings, Memory<bool> tags)
    {
        long start = Stopwatch.GetTimestamp();

        bool[] output = await sentimentInference.BatchPredict(strings);
        var end = Stopwatch.GetElapsedTime(start);

        SysConsole.WriteLine($"elapsed time: {end.TotalSeconds}s");
        SysConsole.WriteLine($"avg time: {end.TotalMilliseconds / strings.Length}ms/it");
        int correct = output.Where((t, i) => t == tags.Span[i]).Count();

        SysConsole.WriteLine($"Correct predictions: {correct}/{output.Length}={correct * 1.0 / output.Length}");
    }

    async Task<(string[] input, bool[] expectedOutput)> LoadTrainingData(string s)
    {
        IList<TrainingData> data = await TrainingParquetReader.ReadParquetFileAsync(s);
        string[] strings = data.Select(x => x.Sentence).ToArray();
        SysConsole.WriteLine($"Parquet file loaded with sentences: {strings.Length}");
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
}