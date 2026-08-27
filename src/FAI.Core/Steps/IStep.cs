namespace FAI.Core.Steps;

public interface IStep<in TInput, in TOutput>
{
    ValueTask ExecuteAsync(TInput input, TOutput output, CancellationToken cancellationToken = default);
}

public interface IAllocatingStep<in TInput, TOutput> : IStep<TInput, TOutput>
{
    ValueTask<BatchLease<TOutput>> RentOutputAsync(TInput input, CancellationToken cancellationToken = default);

    async ValueTask<BatchLease<TOutput>> ExecuteAsync(
        TInput input,
        CancellationToken cancellationToken = default)
    {
        BatchLease<TOutput> output = await RentOutputAsync(input, cancellationToken);
        try
        {
            await ExecuteAsync(input, output.Value, cancellationToken);
            return output;
        }
        catch
        {
            output.Dispose();
            throw;
        }
    }
}
