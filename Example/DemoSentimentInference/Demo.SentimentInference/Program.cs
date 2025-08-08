using Example.SentimentInference.Console;
using Example.SentimentInference.Model;
using FAI.Core.Abstractions;
using FAI.Core.PipelineBatchExecutors;
using FAI.Core.ResultTypes;
using FAI.NLP.Configuration;
using FAI.NLP.InferenceTasks.TextClassification;
using FAI.NLP.PipelineBatchExecutors;
using FAI.NLP.Tokenization;
using FAI.Onnx.Configuration;
using FAI.Onnx.Factories;

var modelDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "ClassificationModelResources");
var evaluationManager = new EvaluationManager(@"D:\Code\NET\ML\fai\Example\SentimentInference\Example.SentimentInference.Console\train-00000-of-00001.parquet");

await evaluationManager.LoadData();
var options = new SentimentInferenceOptions(
    ModelDir: modelDir,
    TokenizerOptions: new PretrainedTokenizerOptions(),
    ModelExecutorType: ModelExecutorType.Simple
);

var executorOptions = new OnnxModelExecutorOptions()
    .ConfigureOnnxOptions(onnxOptions =>
    {
        onnxOptions.ConfigureSessionOptions(sessionOptions =>
        {
            sessionOptions.AppendExecutionProvider_CUDA();
            Console.WriteLine("Using GPU accelerator");

            sessionOptions.AppendExecutionProvider_CPU();
        });
        onnxOptions.ModelDir = options.ModelDir;
    });

Task<PretrainedTokenizer> tokenizer = TokenizationUtils.BERTTokenizerFromPretrained(options.ModelDir, options.TokenizerOptions);
Task<IModelExecutor<long, float>> model = OnnxModelExecutorFactory.CreateModelExecutor(options.ModelExecutorType, executorOptions).AsTask();
await Task.WhenAll(tokenizer, model);

TextClassification<bool> classificationTask = await new TextClassificationBuilder<bool>()
    .UseChoices(false, true)
    .UseTokenizer(tokenizer.Result)
    .UseModelExecutor(model.Result)
    .BuildAsync();

var executor = new SerialPipelineBatchExecutor<TokenizedText, ClassificationResult<bool>>(classificationTask, maxBatchSize: 10);

var pipeline = new Pipeline<TokenizedText, ClassificationResult<bool>>(executor);

var sentimentInference = new SentimentInference(pipeline);

#region predict example

Console.Write(">");
Console.ReadLine();
Console.WriteLine("predict example:");

string sentence = "This cat is a very cute cat!";
bool prediction = await sentimentInference.Predict(sentence);

Console.WriteLine($"Does the sentence: '{sentence}' have a positive sentiment? {prediction}\n");

#endregion

#region batch prediction

Console.Write(">");
Console.ReadLine();
Console.WriteLine("batch predict example:");

string[] sentences =
[
    "This cat is a very cute cat!",
    "This dog is not a good dog.",
    "I love this movie, it's fantastic!",
    "The weather today is terrible."
];

bool[] predictions = await sentimentInference.BatchPredict(sentences);
Console.WriteLine("Batch predictions:");
for (int i = 0; i < sentences.Length; i++)
{
    Console.WriteLine($"Sentence: '{sentences[i]}' - Is Positive? {predictions[i]}");
}

#endregion

#region evaluations

Console.WriteLine(">");
Console.ReadLine();
int sampleSize = 20000;
Console.WriteLine("Sample size: {0}", sampleSize);

#endregion

#region performance evaluation - 1

Console.Write(">");
Console.ReadLine();
Console.WriteLine("evaluation:");

await evaluationManager.Run(sentimentInference, sampleSize);

#endregion

#region stage 2

Console.Write(">");
Console.ReadLine();
Console.WriteLine("parallel evaluation:");

var streamedBatchExecutor = new StreamedBatchExecutor<TokenizedText, BatchTokenizedResult, ClassificationResult<bool>[], ClassificationResult<bool>>(
    classificationTask,
    maxBatchSize: 10,
    maxConcurrency: 4,
    parallelTokenization: false);

pipeline = new Pipeline<TokenizedText, ClassificationResult<bool>>(streamedBatchExecutor);

sentimentInference = new SentimentInference(pipeline);

await evaluationManager.Run(sentimentInference, sampleSize);

#endregion

#region stage 2.1

Console.Write(">");
Console.ReadLine();
Console.WriteLine("parallel evaluation (including tokenization):");

streamedBatchExecutor = new StreamedBatchExecutor<TokenizedText, BatchTokenizedResult, ClassificationResult<bool>[], ClassificationResult<bool>>(
    classificationTask,
    maxBatchSize: 10,
    maxConcurrency: 4,
    parallelTokenization: true);

pipeline = new Pipeline<TokenizedText, ClassificationResult<bool>>(streamedBatchExecutor);

sentimentInference = new SentimentInference(pipeline);

await evaluationManager.Run(sentimentInference, sampleSize);

#endregion

#region stage 3

Console.Write(">");
Console.ReadLine();
Console.WriteLine("sorted by token count:");

streamedBatchExecutor = new StreamedBatchExecutor<TokenizedText, BatchTokenizedResult, ClassificationResult<bool>[], ClassificationResult<bool>>(
    classificationTask,
    maxBatchSize: 10,
    maxConcurrency: 4,
    parallelTokenization: false);

var tokenSortingExecutor = new TokenCountSortingBatchExecutor<TokenizedText, ClassificationResult<bool>>(streamedBatchExecutor, tokenizer.Result);

pipeline = new Pipeline<TokenizedText, ClassificationResult<bool>>(tokenSortingExecutor);

sentimentInference = new SentimentInference(pipeline);
await evaluationManager.Run(sentimentInference, sampleSize);

#endregion

#region stage 4

Console.Write(">");
Console.ReadLine();
Console.WriteLine("sorted by token count:");

streamedBatchExecutor = new StreamedBatchExecutor<TokenizedText, BatchTokenizedResult, ClassificationResult<bool>[], ClassificationResult<bool>>(
    classificationTask,
    maxBatchSize: null,
    maxConcurrency: 4,
    parallelTokenization: false);

var paddedTokensBatchExecutor = new MaxPaddedTokensBatchExecutor<TokenizedText, ClassificationResult<bool>>(
    streamedBatchExecutor,
    maxPaddedTokenRatio: 0.1,
    maxTokenCount: 2048);

tokenSortingExecutor = new TokenCountSortingBatchExecutor<TokenizedText, ClassificationResult<bool>>(paddedTokensBatchExecutor, tokenizer.Result);

pipeline = new Pipeline<TokenizedText, ClassificationResult<bool>>(tokenSortingExecutor);
sentimentInference = new SentimentInference(pipeline);
await evaluationManager.Run(sentimentInference, sampleSize);

#endregion