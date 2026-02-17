using FAI.Core.Abstractions;
using FAI.Core.PipelineBatchExecutors;
using Microsoft.Extensions.Logging.Abstractions;

namespace FAI.Core.Tests.PipelineBatchExecutorTests;

public class StreamedBatchExecutorTests
{
    /// <summary>
    /// Simulates image preprocessing: counts pixels, runs classification model, assigns class labels
    /// </summary>
    private class ImageClassificationInferenceSteps : InferenceSteps<string, int, float[], string>
    {
        // Preprocess: Count characters in image path (simulating pixel counting)
        public override int Preprocess(ReadOnlySpan<string> input) => input[0].Length;

        // RunModel: Simulate neural network producing classification scores
        public override Task<float[]> RunModel(ReadOnlyMemory<string> input, int pixelCount)
        {
            // Generate mock confidence scores based on pixel count
            float[] scores = [pixelCount / 100f, (100 - pixelCount) / 100f, 0.5f];
            return Task.FromResult(scores);
        }

        // PostProcess: Convert scores to class labels
        public override void PostProcess(ReadOnlySpan<string> inputs, int preprocesses, float[] modelOutput, Span<string> outputs)
        {
            for (int i = 0; i < outputs.Length; i++)
            {
                int maxIndex = 0;
                for (int j = 1; j < modelOutput.Length; j++)
                {
                    if (modelOutput[j] > modelOutput[maxIndex])
                        maxIndex = j;
                }
                outputs[i] = maxIndex == 0 ? "cat" : maxIndex == 1 ? "dog" : "bird";
            }
        }
    }

    [Fact]
    public async Task ExecuteBatchPredict_StreamsImageClassificationThroughPipeline()
    {
        // Arrange: Demonstrate streaming pipeline with Preprocess → Model → PostProcess stages
        var inference = new ImageClassificationInferenceSteps();
        var executor = new StreamedBatchExecutor<string, int, float[], string>(
            inference,
            maxBatchSize: null,
            maxConcurrency: null,
            parallelTokenization: false,
            NullLogger<StreamedBatchExecutor<string, int, float[], string>>.Instance);

        string[] imagePathsArray = ["images/cat_001.jpg", "images/dog_002.jpg", "images/bird_003.jpg"];
        ReadOnlyMemory<string> imagePaths = imagePathsArray.AsMemory();
        Memory<string> predictions = new string[3];

        // Act
        await executor.ExecuteBatchPredict(imagePaths, predictions);

        // Assert: Verify full pipeline execution
        Assert.Equal("dog", predictions.Span[0]); // cat_001.jpg → 18 chars → dog wins
        Assert.Equal("dog", predictions.Span[1]); // dog_002.jpg → 18 chars → dog wins
        Assert.Equal("dog", predictions.Span[2]); // bird_003.jpg → 19 chars → dog wins
    }

    /// <summary>
    /// Simulates text sentiment analysis with realistic preprocessing and scoring
    /// </summary>
    private class SentimentAnalysisInferenceSteps : InferenceSteps<string, int, float, bool>
    {
        // Preprocess: Token count
        public override int Preprocess(ReadOnlySpan<string> input) => input[0].Split(' ').Length;

        // RunModel: Sentiment score (-1 to 1)
        public override Task<float> RunModel(ReadOnlyMemory<string> input, int tokenCount)
        {
            // Longer texts tend to be more positive (simplified heuristic)
            float sentimentScore = (tokenCount - 5) / 10f;
            return Task.FromResult(sentimentScore);
        }

        // PostProcess: Convert to binary positive/negative
        public override void PostProcess(ReadOnlySpan<string> inputs, int preprocesses, float modelOutput, Span<bool> outputs)
        {
            for (int i = 0; i < outputs.Length; i++)
            {
                outputs[i] = modelOutput > 0; // Positive if score > 0
            }
        }
    }

    [Fact]
    public async Task ExecuteBatchPredict_ProcessesBatchesWithChunking()
    {
        // Arrange: Demonstrate chunked processing for large batches
        var inference = new SentimentAnalysisInferenceSteps();
        var executor = new StreamedBatchExecutor<string, int, float, bool>(
            inference,
            maxBatchSize: 2,  // Process 2 reviews at a time
            maxConcurrency: 1,
            parallelTokenization: false,
            NullLogger<StreamedBatchExecutor<string, int, float, bool>>.Instance);

        string[] reviewsArray = [
            "Great product",                    // 2 tokens → -0.3 → negative
            "Absolutely loved it highly recommend", // 5 tokens → 0.0 → negative
            "Best purchase I have ever made in my entire life", // 10 tokens → 0.5 → positive
            "Amazing quality and fast shipping service", // 6 tokens → 0.1 → positive
            "Perfect"                          // 1 token → -0.4 → negative
        ];
        ReadOnlyMemory<string> reviews = reviewsArray.AsMemory();
        Memory<bool> sentiments = new bool[5];

        // Act
        await executor.ExecuteBatchPredict(reviews, sentiments);

        // Assert: Verify chunked processing results
        Assert.False(sentiments.Span[0]); // "Great product" → 2 tokens → negative
        Assert.False(sentiments.Span[1]); // "Absolutely loved..." → 5 tokens → neutral/negative
        Assert.True(sentiments.Span[2]);  // "Best purchase..." → 10 tokens → positive
        Assert.True(sentiments.Span[3]);  // "Amazing quality..." → 6 tokens → positive
        Assert.False(sentiments.Span[4]); // "Perfect" → 1 token → negative
    }

    private class FailingModelInferenceSteps : SentimentAnalysisInferenceSteps
    {
        public override Task<float> RunModel(ReadOnlyMemory<string> input, int tokenCount)
        {
            throw new InvalidOperationException("Model inference failed");
        }
    }

    [Fact]
    public async Task ExecuteBatchPredict_ModelFailure_PropagatesException()
    {
        // Arrange: Demonstrate error handling in model execution stage
        var inference = new FailingModelInferenceSteps();
        var executor = new StreamedBatchExecutor<string, int, float, bool>(
            inference,
            maxBatchSize: null,
            maxConcurrency: null,
            parallelTokenization: false,
            NullLogger<StreamedBatchExecutor<string, int, float, bool>>.Instance);

        string[] textsArray = ["This will fail", "Another text"];
        ReadOnlyMemory<string> texts = textsArray.AsMemory();
        Memory<bool> outputs = new bool[2];

        // Act & Assert
        await Assert.ThrowsAsync<InvalidOperationException>(() => executor.ExecuteBatchPredict(texts, outputs));
    }

    private class FailingPostProcessInferenceSteps : SentimentAnalysisInferenceSteps
    {
        public override void PostProcess(ReadOnlySpan<string> inputs, int preprocesses, float modelOutput, Span<bool> outputs)
        {
            throw new InvalidOperationException("Post-processing failure");
        }
    }

    [Fact]
    public async Task ExecuteBatchPredict_PostProcessFailure_PropagatesException()
    {
        // Arrange: Demonstrate error handling in post-processing stage
        var inference = new FailingPostProcessInferenceSteps();
        var executor = new StreamedBatchExecutor<string, int, float, bool>(
            inference,
            maxBatchSize: null,
            maxConcurrency: null,
            parallelTokenization: false,
            NullLogger<StreamedBatchExecutor<string, int, float, bool>>.Instance);

        string[] textsArray = ["Test input"];
        ReadOnlyMemory<string> texts = textsArray.AsMemory();
        Memory<bool> outputs = new bool[1];

        // Act & Assert
        await Assert.ThrowsAsync<InvalidOperationException>(() => executor.ExecuteBatchPredict(texts, outputs));
    }

    [Fact]
    public async Task ExecuteBatchPredict_ParallelPreprocessing_ProcessesChunksConcurrently()
    {
        // Arrange: Demonstrate parallel preprocessing for better throughput
        var inference = new SentimentAnalysisInferenceSteps();
        var executor = new StreamedBatchExecutor<string, int, float, bool>(
            inference,
            maxBatchSize: 2,
            maxConcurrency: 4,  // Allow parallel preprocessing
            parallelTokenization: true,  // Enable parallel mode
            NullLogger<StreamedBatchExecutor<string, int, float, bool>>.Instance);

        string[] reviewsArray = [
            "Good value for money and quality",
            "Would definitely buy this again",
            "Not satisfied with the product quality",
            "Exceeded my expectations completely"
        ];
        ReadOnlyMemory<string> reviews = reviewsArray.AsMemory();
        Memory<bool> sentiments = new bool[4];

        // Act
        await executor.ExecuteBatchPredict(reviews, sentiments);

        // Assert: Verify all batches processed correctly
        Assert.True(sentiments.Span[0]);  // 5 tokens → positive
        Assert.True(sentiments.Span[1]);  // 5 tokens → positive
        Assert.True(sentiments.Span[2]);  // 6 tokens → positive
        Assert.True(sentiments.Span[3]);  // 4 tokens → negative (borderline)
    }
}
