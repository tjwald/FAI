namespace FAI.Evaluation;

public interface IEvaluator<TInferenceInput, TInferenceOutput, TEvaluationResult>
{
    Task<TEvaluationResult> Evaluate(IAsyncEnumerable<(TInferenceInput[], TInferenceOutput[])> inferenceResults);
}
