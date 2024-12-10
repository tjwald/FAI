namespace ML.Infra.Abstractions;


public abstract class Pipeline<TInput, TOutput, TPreprocess, TModelOutput>
{
    private readonly IPipelineBatchExecutor<TInput, TOutput, TPreprocess, TModelOutput> _executor;

    protected Pipeline(IPipelineBatchExecutor<TInput, TOutput, TPreprocess, TModelOutput> executor)
    {
        _executor = executor;
    }

    public async Task<TOutput> Predict(TInput input)
    {
        TInput[] inputArr = [input];
        return (await BatchPredict(inputArr))[0];
    }

    public async Task<TOutput[]> BatchPredict(ReadOnlyMemory<TInput> inputs)
    {
        var outputs = new TOutput[inputs.Length];
        Memory<TOutput> outputSpan = outputs.AsMemory();

        await _executor.ExecuteBatchPredict(this, inputs, outputSpan);

        return outputs;
    }

    internal async Task ProcessBatch(ReadOnlyMemory<TInput> inputs, Memory<TOutput> outputs)
    {
        var preprocess = Preprocess(inputs.Span);
        var modelOutput = await RunModel(inputs, preprocess);
        PostProcess(inputs.Span, preprocess, modelOutput, outputs.Span);
    }

    protected abstract TPreprocess Preprocess(ReadOnlySpan<TInput> input);
    
    protected abstract Task<TModelOutput> RunModel(ReadOnlyMemory<TInput> input, TPreprocess preprocesses);
    protected abstract void PostProcess(ReadOnlySpan<TInput> inputs, TPreprocess preprocesses, TModelOutput modelOutput, Span<TOutput> outputs);
}