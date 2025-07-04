using System.Diagnostics;
using System.Threading.Channels;
using FAI.Core.Abstractions;

namespace FAI.Core;

/// <summary>
/// Orchestrates inference operations by batching requests and managing concurrency for dynamic requests.
/// </summary>
/// <typeparam name="TInference">The type of the inference model implementing <see cref="IInference{TInput,TOutput}"/>.</typeparam>
/// <typeparam name="TQuery">The type of the input query for inference.</typeparam>
/// <typeparam name="TResult">The type of the result produced by the inference.</typeparam>
public sealed class InferenceOrchestrator<TInference, TQuery, TResult> : IInference<TQuery, TResult> where TInference : IInference<TQuery, TResult>
{
    private readonly Lazy<TInference> _modelInstance;
    private readonly int _maxBatchSize;
    private readonly TimeSpan _emptyQueueSleepDuration;
    private readonly Channel<(TQuery, TaskCompletionSource<TResult>, ActivityContext?)> _queue;
    private readonly SemaphoreSlim _semaphore;
    private static readonly ActivitySource ActivitySource = new ActivitySource("ModelPredictionOrchestrator");

    /// <summary>
    /// Initializes a new instance of the <see cref="InferenceOrchestrator{TInference, TQuery, TResult}"/> class.
    /// </summary>
    /// <param name="modelInstance">A lazy-loaded instance of the inference model.</param>
    /// <param name="maxBatchSize">The maximum number of requests to process in a single batch.</param>
    /// <param name="maxConcurrentBatches">The maximum number of concurrent batches allowed.</param>
    /// <param name="emptyQueueSleepDuration">The duration to wait when the queue is empty before checking again.</param>
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

    /// <summary>
    /// Predicts the result for a single input query asynchronously.
    /// </summary>
    /// <param name="inputQuery">The input query for the prediction.</param>
    /// <returns>A task representing the asynchronous operation. The task result contains the predicted result.</returns>
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

    /// <summary>
    /// Starts the background processing task to handle batched inference requests.
    /// </summary>
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

    /// <summary>
    /// Releases the semaphore after a batch task completes.
    /// </summary>
    /// <param name="t">The completed task.</param>
    private void ReleaseSemaphore(Task t) => _semaphore.Release();

    /// <summary>
    /// Processes a batch of inference requests dynamically.
    /// </summary>
    /// <param name="model">The inference model to use for processing the batch.</param>
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

    /// <summary>
    /// Retrieves available requests from the queue up to the specified maximum count.
    /// </summary>
    /// <param name="maxCount">The maximum number of requests to retrieve.</param>
    /// <returns>A list of requests retrieved from the queue.</returns>
    private List<(TQuery, TaskCompletionSource<TResult>, ActivityContext?)> GetAvailableRequestsAsync(int maxCount)
    {
        var requests = new List<(TQuery, TaskCompletionSource<TResult>, ActivityContext?)>();
        while (requests.Count < maxCount && _queue.Reader.TryRead(out var item))
        {
            requests.Add(item);
        }

        return requests;
    }

    /// <summary>
    /// Predicts the results for a batch of input queries asynchronously.
    /// </summary>
    /// <param name="input">A read-only memory containing the batch of input queries.</param>
    /// <returns>A task representing the asynchronous operation. The task result contains an array of predicted results.</returns>
    public Task<TResult[]> BatchPredict(ReadOnlyMemory<TQuery> input) => _modelInstance.Value.BatchPredict(input);
}