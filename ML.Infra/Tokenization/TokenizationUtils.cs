using Microsoft.ML.Tokenizers;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace ML.Infra.Tokenization;

public static class TokenizationUtils
{
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

    public static async Task<PretrainedTokenizer> BERTTokenizerFromPretrained(string path, PretrainedTokenizerOptions tokenizerOptions)
    {
        var streamVocab = File.OpenRead(Path.Combine(path, "vocab.txt"));
        var streamConfig = File.OpenRead(Path.Combine(path, "tokenizer_config.json"));
        var config = JsonSerializer.Deserialize(streamConfig, TokenizationOptionsJsonSerializerContext.Default.BertOptions);
        Console.WriteLine(config);
        var tokenizer = await BertTokenizer.CreateAsync(streamVocab, config);
        return new PretrainedTokenizer(tokenizer, tokenizerOptions);
    }
}


[JsonSerializable(typeof(BertOptions))]
[JsonSerializable(typeof(Dictionary<string, int>))]
public partial class TokenizationOptionsJsonSerializerContext : JsonSerializerContext
{
    
}