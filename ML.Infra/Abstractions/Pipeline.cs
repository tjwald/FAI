namespace ML.Infra.Abstractions;

public abstract class Pipeline<TInput, TOutput, TPreprocess, TBatchPreprocess, TModelOutput>
{
    private readonly int _optimalBatchSize;

    protected Pipeline(int optimalBatchSize)
    {
        _optimalBatchSize = optimalBatchSize;
    }

    public async Task<TOutput> Predict(TInput input)
    {
        return (await BatchPredict([input]))[0];
    }

    public async Task<TOutput[]> BatchPredict(TInput[] inputs)
    {
        //Stopwatch stopwatch = Stopwatch.StartNew();
        TBatchPreprocess preprocesses = await Preprocess(inputs);
        //stopwatch.Stop();
        //Console.WriteLine($"Preprocessing: {stopwatch.Elapsed}");


        TOutput[] outputs = new TOutput[inputs.Length];
        int i = 0;
        //stopwatch.Restart();
        foreach (var (batchSize, preprocessBatch) in BatchPreprocesses(preprocesses, _optimalBatchSize))
        {
            var inputsSlice = inputs.AsMemory(i, batchSize);
            TModelOutput modelOutput = await RunModel(inputsSlice, preprocessBatch);
            PostProcess(inputsSlice.Span, preprocessBatch, modelOutput, outputs.AsSpan(i, batchSize));
            i += batchSize;

        }
        //stopwatch.Stop();
        //Console.WriteLine($"Processing: {stopwatch.Elapsed}");
        return outputs;
    }

    protected abstract ValueTask<TBatchPreprocess> Preprocess(ReadOnlyMemory<TInput> input);

    protected abstract IEnumerable<(int, TPreprocess)> BatchPreprocesses(TBatchPreprocess preprocesses, int batchSize);

    protected abstract Task<TModelOutput> RunModel(ReadOnlyMemory<TInput> input, TPreprocess preprocesses);
    protected abstract void PostProcess(ReadOnlySpan<TInput> inputs, TPreprocess preprocesses, TModelOutput modelOutput, Span<TOutput> outputs);
}
