using FAI.Core.Configurations.PipelineBatchExecutors;
using FAI.NLP.Tokenization;

namespace FAI.NLP.Configuration.PipelineBatchExecutors;

public abstract class TokenPipelineBatchExecutorBuilder<TToken, TOutput, TSelf>
    : DecoratorExecutorBuilder<TToken, TOutput, TSelf>
    where TToken : ITokenizable
    where TSelf : TokenPipelineBatchExecutorBuilder<TToken, TOutput, TSelf>
{
    private PretrainedTokenizer? _tokenizer;
    private Func<ValueTask<PretrainedTokenizer>>? _tokenizerFactory;

    public TSelf UseTokenizer(PretrainedTokenizer tokenizer)
    {
        _tokenizer = tokenizer;
        return (TSelf)this;
    }

    public TSelf UseTokenizer(Func<ValueTask<PretrainedTokenizer>> tokenizerFactory)
    {
        _tokenizerFactory = tokenizerFactory;
        return (TSelf)this;
    }

    protected async ValueTask<PretrainedTokenizer> GetTokenizer()
    {
        if (_tokenizer is not null)
        {
            return _tokenizer;
        }

        _tokenizer = await _tokenizerFactory!();
        return _tokenizer;
    }
}
