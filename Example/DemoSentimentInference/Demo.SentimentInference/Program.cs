using Example.SentimentInference.Model;
using FAI.Core.Abstractions;
using FAI.Core.PipelineBatchExecutors;
using FAI.Core.ResultTypes;
using FAI.NLP.Configuration;
using FAI.NLP.InferenceTasks.TextClassification;
using FAI.NLP.Tokenization;
using FAI.Onnx.Configuration;
using FAI.Onnx.Factories;

var modelDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "ClassificationModelResources");
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
Task<IModelExecutor<long, float>> model = ModelExecutorFactory.CreateModelExecutor(options.ModelExecutorType, executorOptions).AsTask();
await Task.WhenAll(tokenizer, model);

TextClassification<bool> classificationTask = await new TextClassificationBuilder<bool>()
    .UseChoices(false, true)
    .UseTokenizer(tokenizer.Result)
    .UseModelExecutor(model.Result)
    .BuildAsync();

var executor = new SerialPipelineBatchExecutor<TokenizedText, ClassificationResult<bool>>(classificationTask, maxBatchSize: 10);

var pipeline = new Pipeline<TokenizedText, ClassificationResult<bool>>(executor);

var sentimentInference = new SentimentInference(pipeline);

string sentence = "This cat is a very cute cat!";
bool prediction = await sentimentInference.Predict(sentence);

Console.WriteLine($"Does the sentence: '{sentence}' have a positive sentiment? {prediction}\n");


// string[] sentences = [
//     "This cat is a very cute cat!",
//     "This dog is not a good dog.",
//     "I love this movie, it's fantastic!",
//     "The weather today is terrible."
// ];

// bool[] predictions = await sentimentInference.BatchPredict(sentences);
// Console.WriteLine("Batch predictions:");
// for (int i = 0; i < sentences.Length; i++)
// {
//     Console.WriteLine($"Sentence: '{sentences[i]}' - Is Positive? {predictions[i]}");
// }


// var streamedBatchExecutor = new StreamedBatchExecutor<TokenizedText, BatchTokenizedResult, ClassificationResult<bool>[], ClassificationResult<bool>>(
//     classificationTask,
//     maxBatchSize: null,
//     maxConcurrency: 4,
//     parallelTokenization: false);

// var paddedTokensBatchExecutor = new MaxPaddedTokensBatchExecutor<TokenizedText, ClassificationResult<bool>>(
//     streamedBatchExecutor,
//     maxPaddedTokenRatio: 0.1,
//     maxTokenCount: 2048);

// var executor = new TokenCountSortingBatchExecutor<TokenizedText, ClassificationResult<bool>>(paddedTokensBatchExecutor, tokenizer.Result);