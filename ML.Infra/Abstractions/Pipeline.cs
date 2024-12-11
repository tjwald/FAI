namespace ML.Infra.Abstractions;

public abstract class Pipeline<TInput, TOutput, TPreprocess, TModelOutput>: IPipeline<TInput, TOutput>
{
    private readonly IPipelineBatchExecutor<TInput, TOutput> _executor;

    protected Pipeline(IPipelineBatchExecutor<TInput, TOutput> executor)
    {
        _executor = executor;
    }

    public async Task<TOutput> Predict(TInput input)
    {
        TInput[] inputArr = [input];
        var output = new TOutput[1];
        await ((IPipeline<TInput, TOutput>)this).ProcessBatch(inputArr, output);
        return output[0];
    }

    public async Task<TOutput[]> BatchPredict(ReadOnlyMemory<TInput> inputs)
    {
        var outputs = new TOutput[inputs.Length];
        Memory<TOutput> outputSpan = outputs.AsMemory();

        await _executor.ExecuteBatchPredict(this, inputs, outputSpan);

        return outputs;
    }

    async Task IPipeline<TInput, TOutput>.ProcessBatch(ReadOnlyMemory<TInput> inputs, Memory<TOutput> outputs)
    {
        var preprocess = Preprocess(inputs.Span);
        var modelOutput = await RunModel(inputs, preprocess);
        PostProcess(inputs.Span, preprocess, modelOutput, outputs.Span);
    }

    protected abstract TPreprocess Preprocess(ReadOnlySpan<TInput> input);
    
    protected abstract Task<TModelOutput> RunModel(ReadOnlyMemory<TInput> input, TPreprocess preprocesses);
    protected abstract void PostProcess(ReadOnlySpan<TInput> inputs, TPreprocess preprocesses, TModelOutput modelOutput, Span<TOutput> outputs);
}