using FAI.NLP.Tests.Mocks;
using FAI.NLP.Tokenization;

namespace FAI.NLP.Tests.Tokenization;

public class PretrainedTokenizerTests
{
    [Fact]
    public void Tokenize_SingleInput_ReturnsCorrectIds()
    {
        // Arrange
        var tokenizer = DummyTokenizerFactory.Create();
        string text = "hello world";

        // Act
        var ids = tokenizer.Tokenize(text);

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
        var tokenizer = DummyTokenizerFactory.Create(maxTokenLength: 10);
        string[] inputs = ["hello", "hello world"];

        // Act
        var result = tokenizer.BatchTokenize(inputs);

        // Assert
        Assert.Equal(2, result.BatchSize);
        Assert.True(result.MaxTokenCount >= 2);
        Assert.Equal(result.Tokens.Lengths, result.Mask.Lengths);
    }

    [Fact]
    public void BatchTokensToTensors_PadsCorrectly()
    {
        // Arrange
        var tokenizer = DummyTokenizerFactory.Create();
        List<int>[] inputs = [[15], [15, 16]]; // hello, hello world

        // Act
        var result = tokenizer.BatchTokensToTensors(inputs, maxTokenSize: 2);

        // Assert
        Assert.Equal(2, result.BatchSize);
        Assert.Equal(2, result.MaxTokenCount);

        // Row 0: [hello, PAD] -> [15, 0]
        Assert.Equal(15, result.Tokens[0, 0]);
        Assert.Equal(0, result.Tokens[0, 1]);
        Assert.Equal(1, result.Mask[0, 0]);
        Assert.Equal(0, result.Mask[0, 1]);

        // Row 1: [hello, world] -> [15, 16]
        Assert.Equal(15, result.Tokens[1, 0]);
        Assert.Equal(16, result.Tokens[1, 1]);
        Assert.Equal(1, result.Mask[1, 0]);
        Assert.Equal(1, result.Mask[1, 1]);
    }
}
