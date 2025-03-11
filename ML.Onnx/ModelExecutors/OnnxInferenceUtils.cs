using Microsoft.ML.OnnxRuntime;

namespace ML.Onnx.ModelExecutors;

internal static class OnnxInferenceUtils
{
    public static Task<IDisposableReadOnlyCollection<OrtValue>> RunSessionInferenceAsync(InferenceSession session, RunOptions options, OrtValue[] ortValues)
    {
        var tcs = new TaskCompletionSource<IDisposableReadOnlyCollection<OrtValue>>();
        ThreadPool.QueueUserWorkItem(ortValues =>
        {
            try
            {
                IDisposableReadOnlyCollection<OrtValue> x = session.Run(options, session.InputNames, ortValues, session.OutputNames);
                tcs.SetResult(x);
            }
            catch (Exception e)
            {
                tcs.SetException(e);
            }
        }, ortValues, true);
        return tcs.Task;
    }
}