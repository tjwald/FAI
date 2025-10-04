namespace FAI.Extensions.Evaluation;

public interface IEvaluator<TInferenceInput, TInferenceOutput, TEvaluationResult>
{
    Task<TEvaluationResult> Evaluate(IAsyncEnumerable<(TInferenceInput[], TInferenceOutput[])> inferenceResults);
}
