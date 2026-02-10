using FAI.Core.Abstractions;
using FAI.NLP.Tokenization;

namespace FAI.NLP.PipelineBatchExecutors;

public sealed class TokenizerBatchExecutor<TInput, TOutput> : IPipelineBatchExecutor<TInput, TOutput> where TInput : ITokenizable
{
    private readonly IPipelineBatchExecutor<TInput, TOutput> _executor;
    private readonly PretrainedTokenizer _tokenizer;

    public TokenizerBatchExecutor(IPipelineBatchExecutor<TInput, TOutput> executor, PretrainedTokenizer tokenizer)
    {
        _executor = executor;
        _tokenizer = tokenizer;
    }

    public Task ExecuteBatchPredict(ReadOnlyMemory<TInput> inputs, Memory<TOutput> outputSpan)
    {
        foreach (var input in inputs.Span)
        {
            input.Tokenize(_tokenizer);
        }
        return _executor.ExecuteBatchPredict(inputs, outputSpan);
    }
}
