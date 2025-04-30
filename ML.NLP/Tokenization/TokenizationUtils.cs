using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.ML.Tokenizers;
using ML.NLP.Configuration;

namespace ML.NLP.Tokenization;

/// <summary>
/// Provides utility methods for loading pretrained tokenizers.
/// </summary>
public static class TokenizationUtils
{
    /// <summary>
    /// Loads a Byte-Pair Encoding (BPE) tokenizer from a pretrained model directory.
    /// </summary>
    /// <param name="path">The path to the directory containing the tokenizer model files.</param>
    /// <param name="tokenizerOptions">The configuration options for tokenization.</param>
    /// <returns>
    /// A task representing the asynchronous operation, returning a <see cref="PretrainedTokenizer"/> initialized with BPE settings.
    /// </returns>
    public static async Task<PretrainedTokenizer> BpeTokenizerFromPretrained(string path, PretrainedTokenizerOptions tokenizerOptions)
    {
        var streamVocab = File.OpenRead(Path.Combine(path, "vocab.json"));
        Stream? streamMerges;
        try
        {
            streamMerges = File.OpenRead(Path.Combine(path, "merges.txt"));
        }
        catch (FileNotFoundException)
        {
            streamMerges = null;
        }

        var streamAddedTokens = File.OpenRead(Path.Combine(path, "added_tokens.json"));
        var addedTokens = JsonSerializer.Deserialize(streamAddedTokens, TokenizationOptionsJsonSerializerContext.Default.DictionaryStringInt32);

        var tokenizer = await BpeTokenizer.CreateAsync(streamVocab, streamMerges, specialTokens: addedTokens);
        return new PretrainedTokenizer(tokenizer, tokenizerOptions);
    }

    /// <summary>
    /// Loads a BERT tokenizer from a pretrained model directory.
    /// </summary>
    /// <param name="path">The path to the directory containing the tokenizer model files.</param>
    /// <param name="tokenizerOptions">The configuration options for tokenization.</param>
    /// <returns>
    /// A task representing the asynchronous operation, returning a <see cref="PretrainedTokenizer"/> initialized with BERT settings.
    /// </returns>
    public static async Task<PretrainedTokenizer> BERTTokenizerFromPretrained(string path, PretrainedTokenizerOptions tokenizerOptions)
    {
        var streamVocab = File.OpenRead(Path.Combine(path, "vocab.txt"));
        var streamConfig = File.OpenRead(Path.Combine(path, "tokenizer_config.json"));
        var config = JsonSerializer.Deserialize(streamConfig, TokenizationOptionsJsonSerializerContext.Default.BertOptions);
        var tokenizer = await BertTokenizer.CreateAsync(streamVocab, config);
        return new PretrainedTokenizer(tokenizer, tokenizerOptions);
    }

    public static Func<ValueTask<PretrainedTokenizer>> GetTokenizerFactory(this Func<Task<PretrainedTokenizer>> factory)
    {
        StrongBox<PretrainedTokenizer> box = new();
        return async () =>
        {
            box.Value ??= await factory();
            return box.Value;
        };
    }
}

[JsonSerializable(typeof(BertOptions))]
[JsonSerializable(typeof(Dictionary<string, int>))]
public partial class TokenizationOptionsJsonSerializerContext : JsonSerializerContext
{
}