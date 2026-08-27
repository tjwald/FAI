namespace FAI.Core.Steps;

public static class StepExtensions
{
    public static async ValueTask<BatchLease<TOutput>> ExecuteAsync<TInput, TOutput>(
        this IAllocatingStep<TInput, TOutput> step,
        TInput input,
        CancellationToken cancellationToken = default)
        => await step.ExecuteAsync(input, cancellationToken);
}
