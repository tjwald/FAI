namespace FAI.Evaluation;

public interface IInferenceInputGetter<out TInferenceInput>
{
    public TInferenceInput InferenceInput { get; }
}
