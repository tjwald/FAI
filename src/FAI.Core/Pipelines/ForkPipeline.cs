namespace FAI.Core.Pipelines;

public sealed class ForkPipeline<TInput, TBranch> : IPipeline<TInput, (TInput Input, TBranch Output)>
{
    private readonly IPipeline<TInput, TBranch> _branch;

    public ForkPipeline(IPipeline<TInput, TBranch> branch)
    {
        _branch = branch;
    }

    public async ValueTask<(TInput Input, TBranch Output)> ExecuteAsync(
        TInput input,
        CancellationToken cancellationToken = default)
    {
        TBranch branchOutput = await _branch.ExecuteAsync(input, cancellationToken);
        return (input, branchOutput);
    }
}

public sealed class ForkPipeline<TInput, T1, T2> : IPipeline<TInput, (T1 Branch1, T2 Branch2)>
{
    private readonly IPipeline<TInput, T1> _branch1;
    private readonly IPipeline<TInput, T2> _branch2;

    public ForkPipeline(IPipeline<TInput, T1> branch1, IPipeline<TInput, T2> branch2)
    {
        _branch1 = branch1;
        _branch2 = branch2;
    }

    public async ValueTask<(T1 Branch1, T2 Branch2)> ExecuteAsync(
        TInput input,
        CancellationToken cancellationToken = default)
    {
        ValueTask<T1> vt1 = _branch1.ExecuteAsync(input, cancellationToken);
        ValueTask<T2> vt2 = _branch2.ExecuteAsync(input, cancellationToken);

        if (vt1.IsCompletedSuccessfully && vt2.IsCompletedSuccessfully)
        {
            return (vt1.Result, vt2.Result);
        }

        Task<T1> task1 = vt1.AsTask();
        Task<T2> task2 = vt2.AsTask();
        try
        {
            await Task.WhenAll(task1, task2);
            return (task1.Result, task2.Result);
        }
        catch
        {
            if (task1.IsCompletedSuccessfully) await PipelineOutputDisposer.DisposeAsync(task1.Result);
            if (task2.IsCompletedSuccessfully) await PipelineOutputDisposer.DisposeAsync(task2.Result);
            throw;
        }
    }
}
