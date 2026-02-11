using System.Text;
using FAI.Core.Abstractions;
using FAI.Core.Configurations.InferenceTasks;
using FAI.Core.Extensions.DI;
using FAI.Core.ResultTypes;
using FAI.NLP.BatchSlicer;
using FAI.NLP.Configuration;
using FAI.NLP.Configuration.PipelineBatchExecutors;
using FAI.NLP.Extensions.DI;
using FAI.NLP.PipelineBatchExecutors;
using FAI.NLP.Tokenization;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.ML.Tokenizers;

namespace FAI.NLP.Extensions.DI.Tests;

public class BatchExecutorExtensionsTests
{
    private readonly IServiceCollection _services = new ServiceCollection();

    public BatchExecutorExtensionsTests()
    {
        _services.AddSingleton(CreateDummyTokenizer());
    }

    [Fact]
    public void UseTokenSorting_WithExplicitOptions_RegistersExecutor()
    {
        // Arrange
        _services.AddSingleton(Substitute.For<IInferenceSteps<MockTokenizable, int>>());
        var builder = _services.AddPipeline<MockTokenizable, int>();
        var options = new TokenCountSortingBatchExecutorOptions(false);

        // Act
        builder.UseTokenSorting(options);
        var sp = _services.BuildServiceProvider();
        var pipeline = sp.GetRequiredService<IPipeline<MockTokenizable, int>>();

        // Assert
        Assert.NotNull(pipeline);
    }

    [Fact]
    public void UseTokenSorting_WithSection_BindsConfiguration()
    {
        // Arrange
        var configData = new Dictionary<string, string?>
        {
            ["NLP:Sorting:Ascending"] = "false"
        };
        var config = new ConfigurationBuilder().AddInMemoryCollection(configData).Build();
        _services.AddSingleton<IConfiguration>(config);
        _services.AddSingleton(Substitute.For<IInferenceSteps<MockTokenizable, int>>());

        var builder = _services.AddPipeline<MockTokenizable, int>();

        // Act
        builder.UseTokenSorting("NLP:Sorting");
        var sp = _services.BuildServiceProvider();
        var pipeline = sp.GetRequiredService<IPipeline<MockTokenizable, int>>();

        // Assert
        Assert.NotNull(pipeline);
        var options = sp.GetRequiredService<TokenCountSortingBatchExecutorOptions>();
        Assert.False(options.Ascending);
    }

    [Fact]
    public void UseTokenizing_RegistersExecutor()
    {
        // Arrange
        _services.AddSingleton(Substitute.For<IInferenceSteps<MockTokenizable, int>>());
        var builder = _services.AddPipeline<MockTokenizable, int>();

        // Act
        builder.UseTokenizing();
        var sp = _services.BuildServiceProvider();
        var pipeline = sp.GetRequiredService<IPipeline<MockTokenizable, int>>();

        // Assert
        Assert.NotNull(pipeline);
    }

    [Fact]
    public void WithTextClassification_RegistersStepsAndOptions()
    {
        // Arrange
        var configData = new Dictionary<string, string?>
        {
            ["NLP:Classification:Choices:0"] = "Positive",
            ["NLP:Classification:Choices:1"] = "Negative",
            ["NLP:Classification:StoreLogits"] = "true"
        };
        var config = new ConfigurationBuilder().AddInMemoryCollection(configData).Build();
        _services.AddSingleton<IConfiguration>(config);
        _services.AddSingleton(Substitute.For<IModelExecutor<long, float>>());

        var builder = _services.AddPipeline<TokenizedText, ClassificationResult<string, float>>();

        // Act
        builder.WithTextClassification("NLP:Classification");
        var sp = _services.BuildServiceProvider();

        // Assert
        var steps = sp.GetService<IInferenceSteps<TokenizedText, ClassificationResult<string, float>>>();
        Assert.NotNull(steps);
        var options = sp.GetService<ClassificationOptions<string>>();
        Assert.NotNull(options);
        Assert.Equal(["Positive", "Negative"], options.Choices);
        Assert.True(options.StoreLogits);
    }

    [Fact]
    public void WithMaxPaddedTokens_RegistersSlicerAndOptions()
    {
        // Arrange
        var configData = new Dictionary<string, string?>
        {
            ["NLP:Partition:MaxTokenCount"] = "128",
            ["NLP:Partition:MaxPaddedTokenRatio"] = "0.5"
        };
        var config = new ConfigurationBuilder().AddInMemoryCollection(configData).Build();
        _services.AddSingleton<IConfiguration>(config);
        _services.AddSingleton(Substitute.For<IInferenceSteps<MockTokenizable, int>>());

        var builder = _services.AddPipeline<MockTokenizable, int>();

        // Act
        builder.UsePartitioning(p => p.WithMaxPaddedTokens("NLP:Partition"));
        var sp = _services.BuildServiceProvider();

        // Assert
        var slicer = sp.GetService<IBatchSlicer<MockTokenizable>>();
        Assert.NotNull(slicer);
        Assert.IsType<MaxPaddedTokensBatchSlicer<MockTokenizable>>(slicer);
        var options = sp.GetService<MaxPaddedTokensSlicerOptions>();
        Assert.NotNull(options);
        Assert.Equal(128, options.MaxTokenCount);
        Assert.Equal(0.5, options.MaxPaddedTokenRatio);
    }

    public record MockTokenizable(int TokenCount) : ITokenizable
    {
        public int MaxTokenLength => TokenCount;
        public int SentenceCount => 1;
        public void Tokenize(PretrainedTokenizer pretrainedTokenizer) { }
    }

    private static PretrainedTokenizer CreateDummyTokenizer()
    {
        var vocab = new StringBuilder();
        vocab.AppendLine("[PAD]");
        vocab.AppendLine("[CLS]");
        vocab.AppendLine("[SEP]");
        vocab.AppendLine("[MASK]");
        vocab.AppendLine("[UNK]");

        using var ms = new MemoryStream(Encoding.UTF8.GetBytes(vocab.ToString()));
        var bertTokenizer = BertTokenizer.Create(ms);

        var options = new PretrainedTokenizerOptions
        {
            MaxTokenLength = 128,
            PaddingToken = 0,
            TruncationOption = TruncationOption.Longest
        };

        return new PretrainedTokenizer(bertTokenizer, options);
    }
}
