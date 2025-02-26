using System.Threading.Channels;
using ML.Infra.Abstractions;

namespace ML.Infra.PipelineBatchExecutors;

record StreamedInferenceChunk<TInput, TOutput, TPreprocess, TModelOutput>(
    ReadOnlyMemory<TInput> Inputs,
    Memory<TOutput> Outputs,
    TaskCompletionSource CompletionSource,
    TPreprocess PreprocessResult)
{
    internal TModelOutput? ModelResult { get; set; } = default;
}

record ModelExecutionInput<TInput, TOutput, TPreprocess>(
    ReadOnlyMemory<TInput> Inputs,
    Memory<TOutput> Outputs,
    TaskCompletionSource CompletionSource,
    TPreprocess PreprocessResult);

record PostProcessInput<TInput, TOutput, TPreprocess, TModelOutput>(
    ModelExecutionInput<TInput, TOutput, TPreprocess> ModelExecutionInput,
    TModelOutput ModelResult);

public sealed class StreamedBatchExecutor<TInput, TPreprocess, TModelOutput, TOutput> : IPipelineBatchExecutor<TInput, TOutput>
{
    private static readonly UnboundedChannelOptions UnboundedChannelOptions = new()
    {
        AllowSynchronousContinuations = false,
    };

    private readonly InferenceSteps<TInput, TPreprocess, TModelOutput, TOutput> _inference;

    private readonly Channel<StreamedInferenceChunk<TInput, TOutput, TPreprocess, TModelOutput>> _modelInputChannel =
        Channel.CreateUnbounded<StreamedInferenceChunk<TInput, TOutput, TPreprocess, TModelOutput>>(UnboundedChannelOptions);

    private readonly Channel<StreamedInferenceChunk<TInput, TOutput, TPreprocess, TModelOutput>> _postProcessingInputChannel =
        Channel.CreateUnbounded<StreamedInferenceChunk<TInput, TOutput, TPreprocess, TModelOutput>>(UnboundedChannelOptions);

    private readonly Task _modelTask;
    private readonly Task _postProcessingTask;

    private readonly int _maxBatchSize;
    private readonly bool _parallelTokenization;
    private readonly ParallelOptions _parallelOptions;

    public StreamedBatchExecutor(IInferenceSteps<TInput, TOutput> inferenceSteps, int maxBatchSize, int? maxConcurrency, bool parallelTokenization)
    {
        if (inferenceSteps is not InferenceSteps<TInput, TPreprocess, TModelOutput, TOutput> inferenceTaskSteps)
        {
            throw new ArgumentException("Only InferenceSteps<,,,> can be used.", nameof(inferenceSteps));
        }

        _maxBatchSize = maxBatchSize;
        _parallelTokenization = parallelTokenization;
        _inference = inferenceTaskSteps;
        _parallelOptions = maxConcurrency.HasValue ? new ParallelOptions { MaxDegreeOfParallelism = maxConcurrency.Value } : new ParallelOptions();

        _modelTask = BackgroundWorker(_modelInputChannel, _parallelOptions, ModelProcessChunk);
        _postProcessingTask = BackgroundWorker(_postProcessingInputChannel, _parallelOptions, PostProcess);
    }

    public Task ExecuteBatchPredict(ReadOnlyMemory<TInput> inputs, Memory<TOutput> output)
    {
        if (inputs.Length < _maxBatchSize)
        {
            var tcs = new TaskCompletionSource();
            var preprocess = _inference.Preprocess(inputs.Span);
            _modelInputChannel.Writer.TryWrite(new StreamedInferenceChunk<TInput, TOutput, TPreprocess, TModelOutput>(inputs, output, tcs, preprocess));
            return tcs.Task;
        }
        (int batchCountWithoutRemainder, int remainder) = Math.DivRem(inputs.Length, _maxBatchSize);
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

    private void PreprocessParallel(ReadOnlyMemory<TInput> inputs, Memory<TOutput> output, int batchCountWithoutRemainder, int batchCount, Task[] tasks)
    {
        Parallel.For(0, batchCountWithoutRemainder, _parallelOptions, i =>
        {
            var taskCompletionSource = new TaskCompletionSource();
            tasks[i] = taskCompletionSource.Task;
            int startIndex = i * batchCount;
            var r = new Range(startIndex, startIndex + _maxBatchSize);
            var preprocess = _inference.Preprocess(inputs[r].Span);
            _modelInputChannel.Writer.TryWrite(new StreamedInferenceChunk<TInput, TOutput, TPreprocess, TModelOutput>(inputs[r], output[r], taskCompletionSource, preprocess));
        });

        if (batchCountWithoutRemainder < batchCount)
        {
            var taskCompletionSource = new TaskCompletionSource();
            tasks[^1] = taskCompletionSource.Task;
            var r = new Range(batchCountWithoutRemainder * _maxBatchSize, inputs.Length);
            var preprocess = _inference.Preprocess(inputs[r].Span);
            _modelInputChannel.Writer.TryWrite(new StreamedInferenceChunk<TInput, TOutput, TPreprocess, TModelOutput>(inputs[r], output[r], taskCompletionSource, preprocess));
        }
    }

    private void PreprocessSerial(ReadOnlyMemory<TInput> inputs, Memory<TOutput> output, int batchCountWithoutRemainder, int batchCount, Task[] tasks)
    {
        int i = 0;
        for (; i < batchCountWithoutRemainder; i+= _maxBatchSize)
        {
            var taskCompletionSource = new TaskCompletionSource();
            tasks[i] = taskCompletionSource.Task;
            var r = new Range(i, i + _maxBatchSize);
            var preprocess = _inference.Preprocess(inputs[r].Span);
            _modelInputChannel.Writer.TryWrite(new StreamedInferenceChunk<TInput, TOutput, TPreprocess, TModelOutput>(inputs[r], output[r], taskCompletionSource, preprocess));
        }

        if (i < batchCount)
        {
            var taskCompletionSource = new TaskCompletionSource();
            tasks[i] = taskCompletionSource.Task;
            var r = new Range(i, inputs.Length);
            var preprocess = _inference.Preprocess(inputs[r].Span);
            _modelInputChannel.Writer.TryWrite(new StreamedInferenceChunk<TInput, TOutput, TPreprocess, TModelOutput>(inputs[r], output[r], taskCompletionSource, preprocess));
        }
    }

    private static Task BackgroundWorker<T>(Channel<T> channel, ParallelOptions parallelOptions, Func<T, CancellationToken, ValueTask> func)
    {
        Console.WriteLine($"Starting worker thread for processing: {func.Method.Name}");
        return Parallel.ForEachAsync(channel.Reader.ReadAllAsync(), parallelOptions, func);
    }

    private async ValueTask ModelProcessChunk(StreamedInferenceChunk<TInput, TOutput, TPreprocess, TModelOutput> chunk, CancellationToken ct)
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
    
    ValueTask PostProcess(StreamedInferenceChunk<TInput, TOutput, TPreprocess, TModelOutput> postProcessInput, CancellationToken ct)
    {
        try
        {
            _inference.PostProcess(postProcessInput.Inputs.Span, postProcessInput.PreprocessResult, postProcessInput.ModelResult,
                postProcessInput.Outputs.Span);
            postProcessInput.CompletionSource.SetResult();
        }
        catch (Exception e)
        {
            postProcessInput.CompletionSource.SetException(e);
        }
            
        return ValueTask.CompletedTask;
    }
}