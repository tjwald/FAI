using ML.Infra.Abstractions;
using System.Diagnostics;
using System.Threading.Channels;

namespace ML.Infra;

public class InferenceOrchestrator<TInference, TQuery, TResult> : IInference<TQuery, TResult> where TInference : IInference<TQuery, TResult>
{
    private readonly Lazy<TInference> _modelInstance;
    private readonly int _maxBatchSize;
    private readonly TimeSpan _emptyQueueSleepDuration;
    private readonly Channel<(TQuery, TaskCompletionSource<TResult>, ActivityContext?)> _queue;
    private readonly SemaphoreSlim _semaphore;
    private static readonly ActivitySource ActivitySource = new ActivitySource("ModelPredictionOrchestrator");

    public InferenceOrchestrator(
        Lazy<TInference> modelInstance,
        int maxBatchSize,
        int maxConcurrentBatches,
        TimeSpan emptyQueueSleepDuration)
    {
        _modelInstance = modelInstance;
        _maxBatchSize = maxBatchSize;
        _emptyQueueSleepDuration = emptyQueueSleepDuration;
        _queue = Channel.CreateBounded<(TQuery, TaskCompletionSource<TResult>, ActivityContext?)>(maxBatchSize * maxConcurrentBatches * 3);
        _semaphore = new SemaphoreSlim(maxConcurrentBatches);
        StartBackgroundProcessing();
    }

    public async Task<TResult> Predict(TQuery inputQuery)
    {
        var tcs = new TaskCompletionSource<TResult>();

        // Start a tracing span for enqueuing the request
        using (var activity = ActivitySource.StartActivity("enqueue-prediction-request", ActivityKind.Producer))
        {
            var context = activity?.Context;
            await _queue.Writer.WriteAsync((inputQuery, tcs, context));
        }

        // Start a tracing span for waiting for the response
        using (var activity = ActivitySource.StartActivity("wait-for-prediction-response", ActivityKind.Consumer))
        {
            return await tcs.Task.ConfigureAwait(false);
        }
    }

    private void StartBackgroundProcessing()
    {
        Task.Run(async () =>
        {
            var tasks = new List<Task>();
            var modelInstance = _modelInstance.Value;
            while (true)
            {
                if (_queue.Reader.Count == 0)
                {
                    await Task.Delay(_emptyQueueSleepDuration);
                    continue;
                }
                await _semaphore.WaitAsync();

                var batchTask = RunDynamicBatchAsync(modelInstance).ContinueWith(ReleaseSemaphore);
                tasks.Add(batchTask);

                tasks.RemoveAll(t => t.IsCompleted);
            }
        });
    }

    private void ReleaseSemaphore(Task t) => _semaphore.Release();

    private async Task RunDynamicBatchAsync(IInference<TQuery, TResult> model)
    {
        // Start a tracing span for the batch processing
        using var activity = ActivitySource.StartActivity("orchestrated-predict", ActivityKind.Consumer);

        List<(TQuery, TaskCompletionSource<TResult>, ActivityContext?)> requests = GetAvailableRequestsAsync(_maxBatchSize);
        if (requests.Count == 0) return;

        activity?.SetTag("dynamic_batch_size", requests.Count);

        // Link contexts from each request to the batch span
        foreach (var (_, _, context) in requests)
        {
            if (context.HasValue)
            {
                activity?.AddLink(new ActivityLink(context.Value));
            }
        }

        try
        {
            var queries = requests.Select(r => r.Item1).ToArray();
            var results = await model.BatchPredict(queries).ConfigureAwait(false);

            for (int i = 0; i < results.Length; i++)
            {
                requests[i].Item2.SetResult(results[i]);
            }
        }
        catch (Exception ex)
        {
            foreach (var (_, tcs, _) in requests)
            {
                tcs.SetException(ex);
            }
            activity?.SetStatus(ActivityStatusCode.Error, ex.Message);
        }
    }

    private List<(TQuery, TaskCompletionSource<TResult>, ActivityContext?)> GetAvailableRequestsAsync(int maxCount)
    {
        var requests = new List<(TQuery, TaskCompletionSource<TResult>, ActivityContext?)>();
        while (requests.Count < maxCount && _queue.Reader.TryRead(out var item))
        {
            requests.Add(item);
        }
        return requests;
    }

    public Task<TResult[]> BatchPredict(ReadOnlyMemory<TQuery> input) => _modelInstance.Value.BatchPredict(input);
}


