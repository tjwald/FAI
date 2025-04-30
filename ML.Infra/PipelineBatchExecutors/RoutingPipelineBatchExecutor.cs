using ML.Infra.Abstractions;

namespace ML.Infra.PipelineBatchExecutors;

public record BatchExecutionRoutingResult<TInput, TOutput>(IPipelineBatchExecutor<TInput, TOutput> Executor, List<Range> BatchRanges)
{
    public int TotalCount { get; } = BatchRanges.Sum(x => x.End.Value - x.Start.Value);
}

public interface IBatchExecutionRoutingStrategy<TInput, TOutput>
{
    List<BatchExecutionRoutingResult<TInput, TOutput>> Route(IPipelineBatchExecutor<TInput, TOutput>[] executors, ReadOnlyMemory<TInput> inputs);
}

public class RoutingPipelineBatchExecutor<TInput, TOutput> : IPipelineBatchExecutor<TInput, TOutput>
{
    private readonly IBatchExecutionRoutingStrategy<TInput, TOutput> _routingStrategy;
    private readonly IPipelineBatchExecutor<TInput, TOutput>[] _executors;

    public RoutingPipelineBatchExecutor(IPipelineBatchExecutor<TInput, TOutput>[] executors, IBatchExecutionRoutingStrategy<TInput, TOutput> routingStrategy)
    {
        _executors = executors;
        _routingStrategy = routingStrategy;
    }

    public async Task ExecuteBatchPredict(ReadOnlyMemory<TInput> inputs, Memory<TOutput> outputSpan)
    {
        List<BatchExecutionRoutingResult<TInput, TOutput>> routingResults = _routingStrategy.Route(_executors, inputs);
        await Parallel.ForEachAsync(routingResults, async (routingResult, _) =>
        {
            var inputArray = new TInput[routingResult.TotalCount];
            CopyRangesTo(inputs, routingResult.BatchRanges, inputArray);
            var outputArray = new TOutput[routingResult.TotalCount];
            await routingResult.Executor.ExecuteBatchPredict(inputArray, outputArray);
            OutputToRanges(outputArray, outputSpan, routingResult.BatchRanges);
        });
    }

    private static void CopyRangesTo<T>(ReadOnlyMemory<T> input, List<Range> ranges, Memory<T> destination)
    {
        int index = 0;
        foreach (var range in ranges)
        {
            input[range].CopyTo(destination[index..]);
            (_, int length) = range.GetOffsetAndLength(input.Length);
            index += length;
        }
    }

    private static void OutputToRanges<T>(ReadOnlyMemory<T> input, Memory<T> destination, List<Range> ranges)
    {
        int index = 0;
        foreach (var range in ranges)
        {
            (_, int length) = range.GetOffsetAndLength(destination.Length);
            input[index..(index + length)].CopyTo(destination[range]);
            index += length;
        }
    }
}