using System.Threading.Channels;
using FAI.Core.Abstractions;

namespace FAI.Core.PipelineBatchExecutors;

/// <summary>
/// A batch executor that processes input data in background workers split between the different steps of the task.
/// </summary>
/// <typeparam name="TInput">The type of the input data.</typeparam>
/// <typeparam name="TPreprocess">The type of the preprocessing result.</typeparam>
/// <typeparam name="TModelOutput">The type of the model output.</typeparam>
/// <typeparam name="TOutput">The type of the final output data.</typeparam>
public sealed class StreamedBatchExecutor<TInput, TPreprocess, TModelOutput, TOutput> : IPipelineBatchExecutor<TInput, TOutput>
{
    private static readonly UnboundedChannelOptions UnboundedChannelOptions = new()
    {
        AllowSynchronousContinuations = false,
    };

    private readonly InferenceSteps<TInput, TPreprocess, TModelOutput, TOutput> _inference;

    private readonly Channel<StreamedInferenceChunk> _modelInputChannel = Channel.CreateUnbounded<StreamedInferenceChunk>(UnboundedChannelOptions);
    private readonly Channel<StreamedInferenceChunk> _postProcessingInputChannel = Channel.CreateUnbounded<StreamedInferenceChunk>(UnboundedChannelOptions);

    private readonly Task _modelTask;
    private readonly Task _postProcessingTask;

    private readonly int? _maxBatchSize;
    private readonly bool _parallelTokenization;
    private readonly ParallelOptions _parallelOptions;

    /// <summary>
    /// Initializes a new instance of the <see cref="StreamedBatchExecutor{TInput, TPreprocess, TModelOutput, TOutput}"/> class.
    /// </summary>
    /// <param name="inferenceSteps">The inference steps to be executed.</param>
    /// <param name="maxBatchSize">The maximum size of a batch to process.</param>
    /// <param name="maxConcurrency">The maximum degree of parallelism for processing tasks.</param>
    /// <param name="parallelTokenization">Indicates whether tokenization should be parallelized.</param>
    public StreamedBatchExecutor(InferenceSteps<TInput, TPreprocess, TModelOutput, TOutput> inferenceSteps, int? maxBatchSize, int? maxConcurrency,
        bool parallelTokenization)
    {
        _maxBatchSize = maxBatchSize;
        _parallelTokenization = parallelTokenization;
        _inference = inferenceSteps;
        _parallelOptions = maxConcurrency.HasValue ? new ParallelOptions { MaxDegreeOfParallelism = maxConcurrency.Value } : new ParallelOptions();

        _modelTask = BackgroundWorker(_modelInputChannel, _parallelOptions, ModelProcessChunk);
        _postProcessingTask = BackgroundWorker(_postProcessingInputChannel, _parallelOptions, PostProcess);
    }

    /// <summary>
    /// Executes a batch prediction operation by processing the input data and writing the results to the output memory.
    /// </summary>
    /// <param name="inputs">The input data for the batch prediction.</param>
    /// <param name="output">The memory block where the output data will be written.</param>
    /// <returns>A task that represents the asynchronous operation.</returns>
    public Task ExecuteBatchPredict(ReadOnlyMemory<TInput> inputs, Memory<TOutput> output)
    {
        if (!_maxBatchSize.HasValue || inputs.Length < _maxBatchSize)
        {
            var tcs = new TaskCompletionSource();
            var preprocess = _inference.Preprocess(inputs.Span);
            _modelInputChannel.Writer.TryWrite(new StreamedInferenceChunk(inputs, output, tcs, preprocess));
            return tcs.Task;
        }

        (int batchCountWithoutRemainder, int remainder) = Math.DivRem(inputs.Length, _maxBatchSize.Value);
        int batchCount = batchCountWithoutRemainder + (remainder > 0 ? 1 : 0);
        var tasks = new Task[batchCount];
        if (_parallelTokenization)
        {
            PreprocessParallel(inputs, output, batchCountWithoutRemainder, batchCount, tasks);
        }
        else
        {
            PreprocessSerial(inputs, output, batchCountWithoutRemainder, batchCount, tasks);
        }

        return Task.WhenAll(tasks);
    }

    /// <summary>
    /// Preprocesses the input data in parallel batches and writes the results to the model input channel.
    /// </summary>
    private void PreprocessParallel(ReadOnlyMemory<TInput> inputs, Memory<TOutput> output, int batchCountWithoutRemainder, int batchCount, Task[] tasks)
    {
        int maxBatchSize = _maxBatchSize!.Value;
        Parallel.For(0, batchCountWithoutRemainder, _parallelOptions, i =>
        {
            var taskCompletionSource = new TaskCompletionSource();
            tasks[i] = taskCompletionSource.Task;
            int startIndex = i * maxBatchSize;
            var r = new Range(startIndex, startIndex + maxBatchSize);
            var preprocess = _inference.Preprocess(inputs[r].Span);
            _modelInputChannel.Writer.TryWrite(new StreamedInferenceChunk(inputs[r], output[r], taskCompletionSource, preprocess));
        });

        if (batchCountWithoutRemainder < batchCount)
        {
            var taskCompletionSource = new TaskCompletionSource();
            tasks[^1] = taskCompletionSource.Task;
            var r = new Range(batchCountWithoutRemainder * maxBatchSize, inputs.Length);
            var preprocess = _inference.Preprocess(inputs[r].Span);
            _modelInputChannel.Writer.TryWrite(new StreamedInferenceChunk(inputs[r], output[r], taskCompletionSource, preprocess));
        }
    }

    /// <summary>
    /// Preprocesses the input data serially and writes the results to the model input channel.
    /// </summary>
    private void PreprocessSerial(ReadOnlyMemory<TInput> inputs, Memory<TOutput> output, int batchCountWithoutRemainder, int batchCount, Task[] tasks)
    {
        int i = 0;
        int maxBatchSize = _maxBatchSize!.Value;
        for (; i < batchCountWithoutRemainder; i += maxBatchSize)
        {
            var taskCompletionSource = new TaskCompletionSource();
            tasks[i] = taskCompletionSource.Task;

            var r = new Range(i, i + maxBatchSize);
            var preprocess = _inference.Preprocess(inputs[r].Span);
            _modelInputChannel.Writer.TryWrite(new StreamedInferenceChunk(inputs[r], output[r], taskCompletionSource, preprocess));
        }

        if (i < batchCount)
        {
            var taskCompletionSource = new TaskCompletionSource();
            tasks[i] = taskCompletionSource.Task;
            var r = new Range(i, inputs.Length);
            var preprocess = _inference.Preprocess(inputs[r].Span);
            _modelInputChannel.Writer.TryWrite(new StreamedInferenceChunk(inputs[r], output[r], taskCompletionSource, preprocess));
        }
    }

    /// <summary>
    /// Starts a background worker to process items from the specified channel using the provided function.
    /// </summary>
    private static Task BackgroundWorker<T>(Channel<T> channel, ParallelOptions parallelOptions, Func<T, CancellationToken, ValueTask> func)
    {
        Console.WriteLine($"Starting worker thread for processing: {func.Method.Name}");
        return Parallel.ForEachAsync(channel.Reader.ReadAllAsync(), parallelOptions, func);
    }

    /// <summary>
    /// Processes a chunk of data by running the model and writing the results to the postprocessing channel.
    /// </summary>
    private async ValueTask ModelProcessChunk(StreamedInferenceChunk chunk, CancellationToken ct)
    {
        try
        {
            var result = await _inference.RunModel(chunk.Inputs, chunk.PreprocessResult);
            chunk.ModelResult = result;
            _postProcessingInputChannel.Writer.TryWrite(chunk);
        }
        catch (Exception e)
        {
            chunk.CompletionSource.SetException(e);
        }
    }

    /// <summary>
    /// Postprocesses a chunk of data and writes the final output to the provided memory block.
    /// </summary>
    private ValueTask PostProcess(StreamedInferenceChunk postProcessInput, CancellationToken ct)
    {
        try
        {
            _inference.PostProcess(postProcessInput.Inputs.Span, postProcessInput.PreprocessResult, postProcessInput.ModelResult!,
                postProcessInput.Outputs.Span);
            postProcessInput.CompletionSource.SetResult();
        }
        catch (Exception e)
        {
            postProcessInput.CompletionSource.SetException(e);
        }

        return ValueTask.CompletedTask;
    }

    /// <summary>
    /// Represents a chunk of data to be processed in the inference pipeline.
    /// </summary>
    private sealed record StreamedInferenceChunk(
        ReadOnlyMemory<TInput> Inputs,
        Memory<TOutput> Outputs,
        TaskCompletionSource CompletionSource,
        TPreprocess PreprocessResult)
    {
        /// <summary>
        /// Gets or sets the result of the model execution for this chunk.
        /// </summary>
        internal TModelOutput? ModelResult { get; set; }
    }
}