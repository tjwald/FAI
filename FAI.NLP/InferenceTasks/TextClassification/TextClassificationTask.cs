using System.Numerics.Tensors;
using FAI.Core.Abstractions;
using FAI.Core.Configurations.InferenceTasks;
using FAI.Core.InferenceTasks.Classification;
using FAI.NLP.Tokenization;

namespace FAI.NLP.InferenceTasks.TextClassification;

internal sealed class TokenizerWrapper : IPreprocessor<TokenizedText, BatchTokenizedResult, long>
{
    private readonly PretrainedTokenizer _tokenizer;

    public TokenizerWrapper(PretrainedTokenizer tokenizer)
    {
        _tokenizer = tokenizer;
    }

    public BatchTokenizedResult Preprocess(ReadOnlySpan<TokenizedText> input)
    {
        if (input[0].Tokens is null)
        {
            return _tokenizer.BatchTokenize(new TextView(input));
        }

        (Tensor<long> tokenization, Tensor<long> mask) = _tokenizer.BatchTokensToTensors(new TokensView(input));

        return new BatchTokenizedResult(tokenization, mask);
    }
}

/// <summary>
/// Represents a pipeline for text classification tasks.
/// </summary>
/// <typeparam name="TClassification">The type of classification labels.</typeparam>
public class TextClassification<TClassification>
    : ClassificationTask<TokenizedText, BatchTokenizedResult, long, TClassification, float>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="TextClassification{TClassification}"/> class.
    /// </summary>
    /// <param name="tokenizer">The pretrained tokenizer used for text tokenization.</param>
    /// <param name="modelExecutor">The model executor used for inference.</param>
    /// <param name="textClassificationOptions">The configuration options for text classification.</param>
    public TextClassification(
        PretrainedTokenizer tokenizer, IModelExecutor<long, float> modelExecutor, ClassificationOptions<TClassification> textClassificationOptions)
        : base(new TokenizerWrapper(tokenizer), modelExecutor, textClassificationOptions)
    {
    }
}
