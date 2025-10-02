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
    private Func<ValueTask<PretrainedTokenizer>>? _tokenizerFactory;
    private Func<IModelExecutor<long, float>>? _executorFactory;

    protected Func<IModelExecutor<long, float>> ExecutorFactory
    {
        get => _executorFactory!;
        private set => _executorFactory = value;
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

    public TSelf UseModelExecutor(Func<IModelExecutor<long, float>> executor)
    {
        ExecutorFactory = executor;
        return (TSelf)this;
    }

    public abstract ValueTask<TInference> BuildAsync();
}
