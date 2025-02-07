using System.Threading.Channels;
using ML.Infra.Abstractions;

namespace ML.Infra.PipelineBatchExecutors;

public sealed class StreamedBatchExecutor<TInput, TOutput, TPreprocess, TModelOutput> : IPipelineBatchExecutor<TInput, TOutput>
{
    private static readonly UnboundedChannelOptions UnboundedChannelOptions = new()
    {
        AllowSynchronousContinuations = false,
            
    };
    private readonly int _maxBatchSize;
    private readonly bool _parallelTokenization;
    private readonly ParallelOptions _parallelOptions;

    public StreamedBatchExecutor(int maxBatchSize, int? maxConcurrency, bool parallelTokenization)
    {
        _maxBatchSize = maxBatchSize;
        _parallelTokenization = parallelTokenization;
        _parallelOptions = maxConcurrency.HasValue ? new ParallelOptions { MaxDegreeOfParallelism = maxConcurrency.Value } : new ParallelOptions();
    }

    public Task ExecuteBatchPredict(IPipeline<TInput, TOutput> pipeline, ReadOnlyMemory<TInput> inputs, Memory<TOutput> output)
    {
        if (pipeline is not Pipeline<TInput, TOutput, TPreprocess, TModelOutput> standardPipeline)
        {
            return pipeline.ProcessBatch(inputs, output);
        }
        
        var preprocessChannel = Channel.CreateUnbounded<(Range inputRange, TPreprocess preprocessChunk)>(UnboundedChannelOptions);
        var modelChannel = Channel.CreateUnbounded<(Range inputRange, TPreprocess preprocessChunk, TModelOutput modelOutput)>(UnboundedChannelOptions);
        var preprocessWorker = PreprocessingWorker(preprocessChannel, standardPipeline, inputs);
        var modelWorker = ModelProcessingWorker(preprocessChannel, modelChannel, standardPipeline, inputs);
        var postprocessWorker = PostProcessingWorker(modelChannel, standardPipeline, inputs, output);

        return Task.WhenAll(preprocessWorker, modelWorker, postprocessWorker);
    }

    private async Task PreprocessingWorker(Channel<(Range inputRange, TPreprocess preprocessChunk)> preprocessChannel,
        Pipeline<TInput, TOutput, TPreprocess, TModelOutput> pipeline, ReadOnlyMemory<TInput> inputs)
    {
        if (_parallelTokenization)
        {
            await PreprocessParallel(preprocessChannel, pipeline, inputs);
        }
        else
        {
            int batchCount = inputs.Length / _maxBatchSize;
            Range inputRange;
            TPreprocess preprocess;
            for (int i = 0; i < inputs.Length - _maxBatchSize; i += _maxBatchSize)
            {
                inputRange = new Range(i, i + _maxBatchSize);
                preprocess = pipeline.Preprocess(inputs[inputRange].Span);
                preprocessChannel.Writer.TryWrite((inputRange, preprocess));
            }

            int batchStartIndex = batchCount * _maxBatchSize;
            int batchEndIndex = inputs.Length;
            inputRange = new Range(batchStartIndex, batchEndIndex);
            preprocess = pipeline.Preprocess(inputs[inputRange].Span);
            preprocessChannel.Writer.TryWrite((inputRange, preprocess));
        }

        preprocessChannel.Writer.Complete();
    }

    private async Task PreprocessParallel(Channel<(Range inputRange, TPreprocess preprocessChunk)> preprocessChannel, Pipeline<TInput, TOutput, TPreprocess, TModelOutput> pipeline, ReadOnlyMemory<TInput> inputs)
    {
        int maxBatchSize = _maxBatchSize * 4;
        int batchCount = inputs.Length / maxBatchSize;

        var task = Task.Run(() => Parallel.For(0, batchCount, _parallelOptions, (i, _) =>
        {
            int batchStartIndex = i * maxBatchSize;
            int batchEndIndex = batchStartIndex + maxBatchSize;
            for (int j = batchStartIndex; j < batchEndIndex; j+= _maxBatchSize)
            {
                var inputRange = new Range(j, j + _maxBatchSize);
                var preprocess = pipeline.Preprocess(inputs[inputRange].Span);
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
                var preprocess = pipeline.Preprocess(inputs[inputRange].Span);
                preprocessChannel.Writer.TryWrite((inputRange, preprocess));
            }
        }

        await task;
    }

    private async Task ModelProcessingWorker(
        Channel<(Range inputRange, TPreprocess preprocessChunk)> preprocessChannel,
        Channel<(Range inputRange, TPreprocess preprocessChunk, TModelOutput modelOutput)> modelChannel,
        Pipeline<TInput, TOutput, TPreprocess, TModelOutput> pipeline,
        ReadOnlyMemory<TInput> inputs)
    {
        await Parallel.ForEachAsync(preprocessChannel.Reader.ReadAllAsync(), _parallelOptions, (preprocessedResult, _) => ModelProcessChunk(modelChannel, preprocessedResult, pipeline, inputs));
        
        modelChannel.Writer.Complete();
    }

    private static async ValueTask ModelProcessChunk(Channel<(Range inputRange, TPreprocess preprocessChunk, TModelOutput modelOutput)> modelChannel,
        (Range inputRange, TPreprocess preprocessChunk) preprocessedResult,
        Pipeline<TInput, TOutput, TPreprocess, TModelOutput> pipeline,
        ReadOnlyMemory<TInput> inputs)
    {
        var result = await pipeline.RunModel(inputs[preprocessedResult.inputRange], preprocessedResult.preprocessChunk);
        modelChannel.Writer.TryWrite((preprocessedResult.inputRange, preprocessedResult.preprocessChunk, result));
    }

    private async Task PostProcessingWorker(Channel<(Range inputRange, TPreprocess preprocessChunk, TModelOutput modelOutput)> modelChannel,
        Pipeline<TInput, TOutput, TPreprocess, TModelOutput> pipeline,
        ReadOnlyMemory<TInput> inputs,
        Memory<TOutput> output)
    {
        await Parallel.ForEachAsync(modelChannel.Reader.ReadAllAsync(), _parallelOptions, PostProcess);
        return;

        ValueTask PostProcess((Range inputRange, TPreprocess preprocessChunk, TModelOutput modelOutput) modelResult, CancellationToken ct)
        {
            pipeline.PostProcess(inputs[modelResult.inputRange].Span, modelResult.preprocessChunk, modelResult.modelOutput,
                output[modelResult.inputRange].Span);
            return ValueTask.CompletedTask;
        }
    }
}