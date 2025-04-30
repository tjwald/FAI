namespace ML.Infra.Abstractions;

public interface IInferenceStepsBuilder<TInput, TOutput, TInference> where TInference : IInferenceSteps<TInput, TOutput>
{
    ValueTask<TInference> BuildAsync();
}