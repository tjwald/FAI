using System.Threading.Channels;
using ML.Infra.Abstractions;

namespace ML.Infra.PipelineBatchExecutors;

public sealed class StreamedBatchExecutor<TInput, TPreprocess, TModelOutput, TOutput> : IPipelineBatchExecutor<TInput, TOutput>
{
    private static readonly UnboundedChannelOptions UnboundedChannelOptions = new()
    {
        AllowSynchronousContinuations = false,
    };

    private readonly InferenceSteps<TInput, TPreprocess, TModelOutput, TOutput> _inference;
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
    }

    public Task ExecuteBatchPredict(ReadOnlyMemory<TInput> inputs, Memory<TOutput> output)
    {
        var preprocessChannel = Channel.CreateUnbounded<(Range inputRange, TPreprocess preprocessChunk)>(UnboundedChannelOptions);
        var modelChannel = Channel.CreateUnbounded<(Range inputRange, TPreprocess preprocessChunk, TModelOutput modelOutput)>(UnboundedChannelOptions);
        var preprocessWorker = PreprocessingWorker(preprocessChannel, inputs);
        var modelWorker = ModelProcessingWorker(preprocessChannel, modelChannel, inputs);
        var postprocessWorker = PostProcessingWorker(modelChannel, inputs, output);

        return Task.WhenAll(preprocessWorker, modelWorker, postprocessWorker);
    }

    private async Task PreprocessingWorker(Channel<(Range inputRange, TPreprocess preprocessChunk)> preprocessChannel, ReadOnlyMemory<TInput> inputs)
    {
        if (_parallelTokenization)
        {
            await PreprocessParallel(preprocessChannel, inputs);
        }
        else
        {
            int batchCount = inputs.Length / _maxBatchSize;
            Range inputRange;
            TPreprocess preprocess;
            for (int i = 0; i < inputs.Length - _maxBatchSize; i += _maxBatchSize)
            {
                inputRange = new Range(i, i + _maxBatchSize);
                preprocess = _inference.Preprocess(inputs[inputRange].Span);
                preprocessChannel.Writer.TryWrite((inputRange, preprocess));
            }

            int batchStartIndex = batchCount * _maxBatchSize;
            int batchEndIndex = inputs.Length;
            inputRange = new Range(batchStartIndex, batchEndIndex);
            preprocess = _inference.Preprocess(inputs[inputRange].Span);
            preprocessChannel.Writer.TryWrite((inputRange, preprocess));
        }

        preprocessChannel.Writer.Complete();
    }

    private async Task PreprocessParallel(
        Channel<(Range inputRange, TPreprocess preprocessChunk)> preprocessChannel,
        ReadOnlyMemory<TInput> inputs)
    {
        int maxBatchSize = _maxBatchSize * 4;
        int batchCount = inputs.Length / maxBatchSize;

        var task = Task.Run(() => Parallel.For(0, batchCount, _parallelOptions, (i, _) =>
        {
            int batchStartIndex = i * maxBatchSize;
            int batchEndIndex = batchStartIndex + maxBatchSize;
            for (int j = batchStartIndex; j < batchEndIndex; j += _maxBatchSize)
            {
                var inputRange = new Range(j, j + _maxBatchSize);
                var preprocess = _inference.Preprocess(inputs[inputRange].Span);
                preprocessChannel.Writer.TryWrite((inputRange, preprocess));
            }
        }));

        if (inputs.Length % maxBatchSize > 0)
        {
            int batchStartIndex = batchCount * maxBatchSize;
            int batchEndIndex = inputs.Length;
            for (int j = batchStartIndex; j < batchEndIndex; j += _maxBatchSize)
            {
                var inputRange = new Range(j, Math.Min(j + _maxBatchSize, batchEndIndex));
                var preprocess = _inference.Preprocess(inputs[inputRange].Span);
                preprocessChannel.Writer.TryWrite((inputRange, preprocess));
            }
        }

        await task;
    }

    private async Task ModelProcessingWorker(
        Channel<(Range inputRange, TPreprocess preprocessChunk)> preprocessChannel,
        Channel<(Range inputRange, TPreprocess preprocessChunk, TModelOutput modelOutput)> modelChannel,
        ReadOnlyMemory<TInput> inputs)
    {
        await Parallel.ForEachAsync(preprocessChannel.Reader.ReadAllAsync(), _parallelOptions,
            (preprocessedResult, _) => ModelProcessChunk(modelChannel, preprocessedResult, inputs));

        modelChannel.Writer.Complete();
    }

    private async ValueTask ModelProcessChunk(
        Channel<(Range inputRange, TPreprocess preprocessChunk, TModelOutput modelOutput)> modelChannel,
        (Range inputRange, TPreprocess preprocessChunk) preprocessedResult,
        ReadOnlyMemory<TInput> inputs)
    {
        var result = await _inference.RunModel(inputs[preprocessedResult.inputRange], preprocessedResult.preprocessChunk);
        modelChannel.Writer.TryWrite((preprocessedResult.inputRange, preprocessedResult.preprocessChunk, result));
    }

    private async Task PostProcessingWorker(
        Channel<(Range inputRange, TPreprocess preprocessChunk, TModelOutput modelOutput)> modelChannel,
        ReadOnlyMemory<TInput> inputs,
        Memory<TOutput> output)
    {
        await Parallel.ForEachAsync(modelChannel.Reader.ReadAllAsync(), _parallelOptions, PostProcess);
        return;

        ValueTask PostProcess((Range inputRange, TPreprocess preprocessChunk, TModelOutput modelOutput) modelResult, CancellationToken ct)
        {
            _inference.PostProcess(inputs[modelResult.inputRange].Span, modelResult.preprocessChunk, modelResult.modelOutput,
                output[modelResult.inputRange].Span);
            return ValueTask.CompletedTask;
        }
    }
}