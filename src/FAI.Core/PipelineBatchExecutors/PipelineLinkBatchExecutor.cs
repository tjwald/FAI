using System.Buffers;
using FAI.Core.Abstractions;

namespace FAI.Core.PipelineBatchExecutors;

public sealed class PipelineLinkBatchExecutor<TInput, TNextInput, TOutput> : IPipelineBatchExecutor<TInput, TOutput>
{
    private readonly IPipeline<TNextInput, TOutput> _nextPipeline;
    private readonly Func<TInput, TNextInput> _func;
    private readonly ArrayPool<TNextInput> _inputPool;

    public PipelineLinkBatchExecutor(IPipeline<TNextInput, TOutput> nextPipeline, Func<TInput, TNextInput> func, ArrayPool<TNextInput> inputPool)
    {
        _nextPipeline = nextPipeline;
        _func = func;
        _inputPool = inputPool;
    }

    public async Task ExecuteBatchPredict(ReadOnlyMemory<TInput> inputs, Memory<TOutput> outputSpan)
    {
        TNextInput[] nextOutput = _inputPool.Rent(inputs.Length);
        var inputSpan = inputs.Span;
        for (int i = 0; i < inputSpan.Length; i++)
        {
            nextOutput[i] = _func(inputSpan[i]);
        }
        await _nextPipeline.BatchPredict(nextOutput.AsMemory(0, inputSpan.Length), outputSpan);
        _inputPool.Return(nextOutput);
    }
}
