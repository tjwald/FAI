namespace ML.Infra.Abstractions;

public interface IInferenceSteps<TInput, TOutput>
{
    Task ProcessBatch(ReadOnlyMemory<TInput> inputs, Memory<TOutput> outputs);
}

public abstract class InferenceStepsSteps<TInput, TPreprocess, TModelOutput, TOutput>: IInferenceSteps<TInput, TOutput>
{
    public abstract TPreprocess Preprocess(ReadOnlySpan<TInput> input);
    public abstract Task<TModelOutput> RunModel(ReadOnlyMemory<TInput> input, TPreprocess preprocesses);
    public abstract void PostProcess(ReadOnlySpan<TInput> inputs, TPreprocess preprocesses, TModelOutput modelOutput, Span<TOutput> outputs);

    public async Task ProcessBatch(ReadOnlyMemory<TInput> inputs, Memory<TOutput> outputs)
    {
        var preprocess = Preprocess(inputs.Span);
        var modelOutput = await RunModel(inputs, preprocess);
        PostProcess(inputs.Span, preprocess, modelOutput, outputs.Span);
    }
}