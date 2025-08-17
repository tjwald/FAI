using System.Threading.Channels;
using FAI.Core.Abstractions;

namespace FAI.Core.PipelineBatchExecutors;

internal sealed record BackgroundInput<TInput, TOutput>(ReadOnlyMemory<TInput> Inputs, Memory<TOutput> Outputs, TaskCompletionSource TaskCompletionSource);

public class BackgroundPipelineBatchExecutor<TInput, TOutput> : IPipelineBatchExecutor<TInput, TOutput>
{
    private static readonly UnboundedChannelOptions UnboundedChannelOptions = new()
    {
        AllowSynchronousContinuations = false,
        SingleReader = true
    };

    private readonly Channel<BackgroundInput<TInput, TOutput>> _inputChannel;

    // ReSharper disable once PrivateFieldCanBeConvertedToLocalVariable
    private readonly Task[] _backgroundTasks;
    private readonly IPipelineBatchExecutor<TInput, TOutput> _next;

    public BackgroundPipelineBatchExecutor(IPipelineBatchExecutor<TInput, TOutput> next, int workerCount)
    {
        _next = next;
        _inputChannel = Channel.CreateUnbounded<BackgroundInput<TInput, TOutput>>(UnboundedChannelOptions);
        _backgroundTasks = new Task[workerCount];
        for (int i = 0; i < _backgroundTasks.Length; i++)
        {
            _backgroundTasks[i] = BackgroundWorker();
        }
    }

    public Task ExecuteBatchPredict(ReadOnlyMemory<TInput> inputs, Memory<TOutput> outputSpan)
    {
        TaskCompletionSource source = new TaskCompletionSource();
        _inputChannel.Writer.TryWrite(new BackgroundInput<TInput, TOutput>(inputs, outputSpan, source));
        return source.Task;
    }

    private async Task BackgroundWorker()
    {
        ChannelReader<BackgroundInput<TInput, TOutput>> reader = _inputChannel.Reader;
        await foreach (BackgroundInput<TInput, TOutput> input in reader.ReadAllAsync())
        {
            try
            {
                await _next.ExecuteBatchPredict(input.Inputs, input.Outputs);
                input.TaskCompletionSource.SetResult();
            }
            catch (Exception e)
            {
                input.TaskCompletionSource.SetException(e);
            }
        }
    }
}
