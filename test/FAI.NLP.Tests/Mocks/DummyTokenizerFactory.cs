using System.Text;
using FAI.NLP.Configuration;
using FAI.NLP.Tokenization;
using Microsoft.ML.Tokenizers;

namespace FAI.NLP.Tests.Mocks;

public static class DummyTokenizerFactory
{
    public static PretrainedTokenizer Create(int maxTokenLength = 128)
    {
        // Define a minimal vocabulary for BERT tokenizer
        var vocab = new StringBuilder();
        vocab.AppendLine("[PAD]");
        vocab.AppendLine("[unused0]");
        vocab.AppendLine("[unused1]");
        vocab.AppendLine("[unused2]");
        vocab.AppendLine("[unused3]");
        vocab.AppendLine("[unused4]");
        vocab.AppendLine("[unused5]");
        vocab.AppendLine("[unused6]");
        vocab.AppendLine("[unused7]");
        vocab.AppendLine("[unused8]");
        vocab.AppendLine("[unused9]");
        vocab.AppendLine("[CLS]");
        vocab.AppendLine("[SEP]");
        vocab.AppendLine("[MASK]");
        vocab.AppendLine("[UNK]");
        vocab.AppendLine("hello");
        vocab.AppendLine("world");

        using var ms = new MemoryStream(Encoding.UTF8.GetBytes(vocab.ToString()));
        var bertTokenizer = BertTokenizer.Create(ms);

        var options = new PretrainedTokenizerOptions
        {
            MaxTokenLength = maxTokenLength,
            PaddingToken = 0,
            TruncationOption = TruncationOption.Longest
        };

        return new PretrainedTokenizer(bertTokenizer, options);
    }
}
