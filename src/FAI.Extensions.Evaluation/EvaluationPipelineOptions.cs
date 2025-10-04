namespace FAI.Extensions.Evaluation;

public record EvaluationPipelineOptions(
    int? LoadingChunkSize = null,
    bool ParallelLoading = false,
    bool ParallelEvaluation = false,
    bool PublishEvaluationAsEvent = false);
