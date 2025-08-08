using FAI.Core.Abstractions;
using FAI.NLP.Tokenization;

namespace FAI.NLP.Configuration;

public abstract class TextInferenceStepsBuilder<TToken, TResult, TInference, TSelf>
    : IInferenceStepsBuilder<TToken, TResult, TInference>
    where TToken : ITokenizable
    where TInference : IInferenceSteps<TToken, TResult>
    where TSelf : TextInferenceStepsBuilder<TToken, TResult, TInference, TSelf>
{
    private PretrainedTokenizer? _tokenizer;
    private IModelExecutor<long, float>? _executor;
    private Func<ValueTask<PretrainedTokenizer>>? _tokenizerFactory;
    private Func<ValueTask<IModelExecutor<long, float>>>? _executorFactory;

    protected Func<ValueTask<IModelExecutor<long, float>>> ExecutorFactory
    {
        set => _executorFactory = value;
    }

    protected async ValueTask<IModelExecutor<long, float>> GetExecutorFactory()
    {
        if (_executor is not null)
        {
            return _executor;
        }

        _executor = await _executorFactory!();
        return _executor;
    }

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

    public TSelf UseModelExecutor(Func<ValueTask<IModelExecutor<long, float>>> executor)
    {
        ExecutorFactory = executor;
        return (TSelf)this;
    }

    public TSelf UseModelExecutor(IModelExecutor<long, float> executor)
    {
        _executor = executor;
        return (TSelf)this;
    }

    public abstract ValueTask<TInference> BuildAsync();
}