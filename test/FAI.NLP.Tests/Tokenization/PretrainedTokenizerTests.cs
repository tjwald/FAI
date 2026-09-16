using FAI.NLP.Tests.Mocks;
using FAI.NLP.Tokenization;

namespace FAI.NLP.Tests.Tokenization;

public class PretrainedTokenizerFixture : IDisposable
{
    public PretrainedTokenizer Tokenizer { get; } = DummyTokenizerFactory.Create();

    public void Dispose()
    {
        // Cleanup if needed
    }
}

public class PretrainedTokenizerTests : IClassFixture<PretrainedTokenizerFixture>
{
    private readonly PretrainedTokenizer _tokenizer;

    public PretrainedTokenizerTests(PretrainedTokenizerFixture fixture)
    {
        _tokenizer = fixture.Tokenizer;
    }

    [Fact]
    public void Tokenize_SingleInput_ReturnsCorrectIds()
    {
        // Arrange
        string text = "hello world";

        // Act
        var ids = _tokenizer.Tokenize(text);

        // Assert
        // Based on the dummy vocab:
        // [PAD]=0, [unused0..9]=1..10, [CLS]=11, [SEP]=12, [MASK]=13, [UNK]=14, hello=15, world=16
        Assert.Contains(15, ids); // hello
        Assert.Contains(16, ids); // world
    }

    [Fact]
    public void BatchTokenize_Strings_ReturnsCorrectTensorShape()
    {
        // Arrange
        string[] inputs = ["hello", "hello world"];

        // Act
        var result = _tokenizer.BatchTokenize(inputs);

        // Assert
        Assert.Equal(2, result.BatchSize);
        Assert.True(result.MaxTokenCount >= 2);
        Assert.Equal(result.InputIds.Lengths, result.AttentionMask.Lengths);
    }

    [Fact]
    public void BatchTokensToTensors_PadsCorrectly()
    {
        // Arrange
        List<int>[] inputs = [[15], [15, 16]]; // hello, hello world

        // Act
        var result = _tokenizer.BatchTokensToTensors(inputs, maxTokenSize: 2);

        // Assert
        Assert.Equal(2, result.BatchSize);
        Assert.Equal(2, result.MaxTokenCount);

        // Row 0: [hello, PAD] -> [15, 0]
        Assert.Equal(15, result.InputIds[0, 0]);
        Assert.Equal(0, result.InputIds[0, 1]);
        Assert.Equal(1, result.AttentionMask[0, 0]);
        Assert.Equal(0, result.AttentionMask[0, 1]);

        // Row 1: [hello, world] -> [15, 16]
        Assert.Equal(15, result.InputIds[1, 0]);
        Assert.Equal(16, result.InputIds[1, 1]);
        Assert.Equal(1, result.AttentionMask[1, 0]);
        Assert.Equal(1, result.AttentionMask[1, 1]);
    }
}
