namespace ML.Infra.Abstractions;

public class Pipeline<TInput, TOutput>: IPipeline<TInput, TOutput>
{
    private readonly IPipelineBatchExecutor<TInput, TOutput> _executor;

    public Pipeline(IPipelineBatchExecutor<TInput, TOutput> executor)
    {
        _executor = executor;
    }

    public async Task<TOutput> Predict(TInput input)
    {
        TInput[] inputArr = [input];
        TOutput[] outputArr = await BatchPredict(inputArr);
        return outputArr[0];
    }

    public async Task<TOutput[]> BatchPredict(ReadOnlyMemory<TInput> inputs)
    {
        var outputs = new TOutput[inputs.Length];
        Memory<TOutput> outputSpan = outputs.AsMemory();

        await _executor.ExecuteBatchPredict(inputs, outputSpan);

        return outputs;
    }
}