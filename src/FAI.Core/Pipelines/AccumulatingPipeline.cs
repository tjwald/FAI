using System.Buffers;
using System.Threading.Channels;
using FAI.Core.Abstractions;
using Microsoft.Extensions.Logging;

namespace FAI.Core.Pipelines;

public sealed record AccumulatingPipelineOptions
{
    /// <summary>
    /// Maximum number of items to accumulate before forcing a flush.
    /// </summary>
    public int MaxBatchSize { get; init; }

    /// <summary>
    /// Maximum time to wait for a batch to fill before flushing.
    /// </summary>
    public TimeSpan MaxLatency { get; init; }

    /// <summary>
    /// Maximum number of items allowed in the pending queue before backpressure is applied.
    /// </summary>
    public int? BufferCapacity { get; init; } = null;

    /// <summary>
    /// If true, calls to BatchPredict will be split into individual items and
    /// queued just like single Predict calls.
    ///
    /// Use this if you want massive batches to be throttled/chunked
    /// alongside single requests (Fairness).
    ///
    /// If false (default), BatchPredict bypasses the queue and runs immediately
    /// (Lower latency, but higher risk of resource spiking).
    /// </summary>
    public bool UnpackBatch { get; init; } = false;
}


public class AccumulatingPipeline<TInput, TOutput> : IPipeline<TInput, TOutput>, IDisposable
{
    private readonly IPipelineBatchExecutor<TInput, TOutput> _executor;
    private readonly AccumulatingPipelineOptions _options;
    private readonly IFailedBatchPolicy<TInput, TOutput> _failedBatchPolicy;
    private readonly ILogger _logger;

    private readonly Channel<(TInput Input, TaskCompletionSource<TOutput> Tcs)> _queue;
    private readonly CancellationTokenSource _cts = new();
    private readonly Task _pumpTask;

    public AccumulatingPipeline(
        IPipelineBatchExecutor<TInput, TOutput> executor,
        AccumulatingPipelineOptions options,
        IFailedBatchPolicy<TInput, TOutput> failedBatchPolicy,
        ILogger<AccumulatingPipeline<TInput, TOutput>> logger)
    {
        _executor = executor;
        _options = options;
        _failedBatchPolicy = failedBatchPolicy;
        _logger = logger;

        if (_options.BufferCapacity.HasValue)
        {
            var boundedOptions = new BoundedChannelOptions(_options.BufferCapacity.Value)
            {
                FullMode = BoundedChannelFullMode.Wait,
                SingleReader = true,
                SingleWriter = false
            };
            _queue = Channel.CreateBounded<(TInput, TaskCompletionSource<TOutput>)>(boundedOptions);
        }
        else
        {
            var unboundedOptions = new UnboundedChannelOptions { SingleReader = true, SingleWriter = false };
            _queue = Channel.CreateUnbounded<(TInput, TaskCompletionSource<TOutput>)>(unboundedOptions);
        }

        _pumpTask = Task.Run(ProcessQueueAsync);
    }

    public async Task<TOutput> Predict(TInput input)
    {
        var tcs = new TaskCompletionSource<TOutput>(TaskCreationOptions.RunContinuationsAsynchronously);

        if (!await _queue.Writer.WaitToWriteAsync(_cts.Token))
            throw new InvalidOperationException("Pipeline is shutting down");

        await _queue.Writer.WriteAsync((input, tcs), _cts.Token);
        return await tcs.Task;
    }

    public async Task<TOutput[]> BatchPredict(ReadOnlyMemory<TInput> input)
    {
        if (!_options.UnpackBatch)
        {
            var results = new TOutput[input.Length];
            await _executor.ExecuteBatchPredict(input, results);
            return results;
        }

        var tasks = new Task<TOutput>[input.Length];
        var span = input.Span;

        for (int i = 0; i < input.Length; i++)
        {
            tasks[i] = Predict(span[i]);
        }

        return await Task.WhenAll(tasks);
    }

    public async Task BatchPredict(ReadOnlyMemory<TInput> input, Memory<TOutput> output)
    {
        if (!_options.UnpackBatch)
        {
            await _executor.ExecuteBatchPredict(input, output);
            return;
        }

        var tasks = new Task<TOutput>[input.Length];
        var span = input.Span;

        for (int i = 0; i < input.Length; i++)
        {
            tasks[i] = Predict(span[i]);
        }

        var results = await Task.WhenAll(tasks);

        results.CopyTo(output);
    }

    private async Task ProcessQueueAsync()
    {
        var batchList = new List<(TInput Input, TaskCompletionSource<TOutput> Tcs)>(_options.MaxBatchSize);

        try
        {
            while (await _queue.Reader.WaitToReadAsync(_cts.Token))
            {
                if (!_queue.Reader.TryRead(out var firstItem)) continue;
                batchList.Add(firstItem);

                using var timeoutCts = CancellationTokenSource.CreateLinkedTokenSource(_cts.Token);
                timeoutCts.CancelAfter(_options.MaxLatency);

                try
                {
                    while (batchList.Count < _options.MaxBatchSize)
                    {
                        if (!await _queue.Reader.WaitToReadAsync(timeoutCts.Token)) break;
                        while (batchList.Count < _options.MaxBatchSize && _queue.Reader.TryRead(out var nextItem))
                        {
                            batchList.Add(nextItem);
                        }
                    }
                }
                catch (OperationCanceledException) { }

                if (batchList.Count > 0)
                {
                    await FlushBatchAsync(batchList);
                    batchList.Clear();
                }
            }
        }
        catch (OperationCanceledException)
        {
            _logger.LogInformation("Pipeline pump stopping.");
        }
        catch (Exception ex)
        {
            _logger.LogCritical(ex, "Pipeline pump crashed.");
        }
    }

    private async Task FlushBatchAsync(List<(TInput Input, TaskCompletionSource<TOutput> Tcs)> batch)
    {
        TInput[]? rentedInputs = null;
        TOutput[]? rentedOutputs = null;
        int count = batch.Count;

        try
        {
            rentedInputs = ArrayPool<TInput>.Shared.Rent(count);
            rentedOutputs = ArrayPool<TOutput>.Shared.Rent(count);

            for (int i = 0; i < count; i++)
            {
                rentedInputs[i] = batch[i].Input;
            }

            var inputMemory = rentedInputs.AsMemory(0, count);
            var outputMemory = rentedOutputs.AsMemory(0, count);

            try
            {
                await _executor.ExecuteBatchPredict(inputMemory, outputMemory);
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Batch failed, invoking policy.");
                await _failedBatchPolicy.HandleAsync(inputMemory, outputMemory, _executor, ex, CancellationToken.None);
            }

            for (int i = 0; i < count; i++)
            {
                batch[i].Tcs.SetResult(rentedOutputs[i]);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Critical batch failure.");
            foreach (var item in batch) item.Tcs.TrySetException(ex);
        }
        finally
        {
            if (rentedInputs is not null) ArrayPool<TInput>.Shared.Return(rentedInputs);
            if (rentedOutputs is not null) ArrayPool<TOutput>.Shared.Return(rentedOutputs);
        }
    }

    public void Dispose()
    {
        _queue.Writer.TryComplete();
        _cts.Cancel();
        _cts.Dispose();
    }
}
