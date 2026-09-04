namespace FAI.Core.Pipelines;

public static class AppendedPipeline
{
    public static IPipeline<TInput, TOutput> Create<TInput, TMiddle, TOutput>(
        IPipeline<TInput, TMiddle> previous,
        IPipeline<TMiddle, TOutput> pipeline)
        => pipeline is IDestinationPipeline<TMiddle, TOutput> destinationPipeline
            ? new DestinationAppendedPipeline<TInput, TMiddle, TOutput>(previous, destinationPipeline)
            : new AppendedPipeline<TInput, TMiddle, TOutput>(previous, pipeline);
}

public class AppendedPipeline<TInput, TMiddle, TOutput> : IPipeline<TInput, TOutput>
{
    protected readonly IPipeline<TInput, TMiddle> Previous;
    protected readonly IPipeline<TMiddle, TOutput> Pipeline;

    public AppendedPipeline(IPipeline<TInput, TMiddle> previous, IPipeline<TMiddle, TOutput> pipeline)
    {
        Previous = previous;
        Pipeline = pipeline;
    }

    public async ValueTask<TOutput> ExecuteAsync(TInput input, CancellationToken cancellationToken = default)
    {
        TMiddle intermediate = await Previous.ExecuteAsync(input, cancellationToken);
        try
        {
            return await Pipeline.ExecuteAsync(intermediate, cancellationToken);
        }
        finally
        {
            await PipelineOutputDisposer.DisposeAsync(intermediate);
        }
    }
}

public sealed class DestinationAppendedPipeline<TInput, TMiddle, TOutput>
    : AppendedPipeline<TInput, TMiddle, TOutput>, IDestinationPipeline<TInput, TOutput>
{
    private readonly IDestinationPipeline<TMiddle, TOutput> _destinationPipeline;

    public DestinationAppendedPipeline(
        IPipeline<TInput, TMiddle> previous,
        IDestinationPipeline<TMiddle, TOutput> pipeline)
        : base(previous, pipeline)
    {
        _destinationPipeline = pipeline;
    }

    public async ValueTask ExecuteAsync(TInput input, TOutput destination, CancellationToken cancellationToken = default)
    {
        TMiddle intermediate = await Previous.ExecuteAsync(input, cancellationToken);
        try
        {
            await _destinationPipeline.ExecuteAsync(intermediate, destination, cancellationToken);
        }
        finally
        {
            await PipelineOutputDisposer.DisposeAsync(intermediate);
        }
    }
}
