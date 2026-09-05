namespace FAI.Core.Pipelines;

public static class DestinationPipeline
{
    public static IDestinationPipeline<TInput, TOutput> AsDestinationPipeline<TInput, TOutput>(
        this IPipeline<TInput, TOutput> pipeline,
        IWritableIndexedBatch<TOutput> outputBatch)
    {
        ArgumentNullException.ThrowIfNull(pipeline);
        ArgumentNullException.ThrowIfNull(outputBatch);

        return pipeline is IDestinationPipeline<TInput, TOutput> destinationPipeline
            ? destinationPipeline
            : new BatchCopyDestinationPipeline<TInput, TOutput>(pipeline, outputBatch);
    }

    public static async ValueTask ExecuteAsync<TInput, TOutput>(
        IPipeline<TInput, TOutput> pipeline,
        TInput input,
        TOutput destination,
        IWritableIndexedBatch<TOutput> outputBatch,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(pipeline);
        ArgumentNullException.ThrowIfNull(outputBatch);

        if (pipeline is IDestinationPipeline<TInput, TOutput> destinationPipeline)
        {
            await destinationPipeline.ExecuteAsync(input, destination, cancellationToken);
            return;
        }

        TOutput output = await pipeline.ExecuteAsync(input, cancellationToken);
        try
        {
            outputBatch.Copy(output, destination);
        }
        finally
        {
            await PipelineOutputDisposer.DisposeAsync(output);
        }
    }
}

internal sealed class BatchCopyDestinationPipeline<TInput, TOutput> : IDestinationPipeline<TInput, TOutput>
{
    private readonly IPipeline<TInput, TOutput> _inner;
    private readonly IWritableIndexedBatch<TOutput> _outputBatch;

    public BatchCopyDestinationPipeline(
        IPipeline<TInput, TOutput> inner,
        IWritableIndexedBatch<TOutput> outputBatch)
    {
        _inner = inner;
        _outputBatch = outputBatch;
    }

    public ValueTask<TOutput> ExecuteAsync(TInput input, CancellationToken cancellationToken = default)
        => _inner.ExecuteAsync(input, cancellationToken);

    public async ValueTask ExecuteAsync(TInput input, TOutput destination, CancellationToken cancellationToken = default)
    {
        TOutput output = await _inner.ExecuteAsync(input, cancellationToken);
        try
        {
            _outputBatch.Copy(output, destination);
        }
        finally
        {
            await PipelineOutputDisposer.DisposeAsync(output);
        }
    }
}
