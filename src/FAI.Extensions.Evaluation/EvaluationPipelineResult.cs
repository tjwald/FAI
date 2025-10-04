namespace FAI.Extensions.Evaluation;

public record EvaluationPipelineResult<TEvaluation>(TEvaluation Evaluation, int SampleSize, TimeSpan InferenceRuntime)
{
    public TimeSpan AveragePerSample => InferenceRuntime / SampleSize;
}
