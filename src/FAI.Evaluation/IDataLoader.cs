namespace FAI.Evaluation;

public interface IDataLoader<in TLoaderInput, out TLoadedInput, TInferenceInput>
    where TLoadedInput : IInferenceInputGetter<TInferenceInput>
{
    IAsyncEnumerable<TLoadedInput> LoadData(TLoaderInput args);
}
