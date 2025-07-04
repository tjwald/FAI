using Microsoft.ML.OnnxRuntime;

namespace FAI.Onnx.Utils;

/// <summary>
/// Provides utility methods for performing asynchronous inference using ONNX runtime sessions.
/// </summary>
internal static class OnnxInferenceUtils
{
    /// <summary>
    /// Runs inference asynchronously on the provided ONNX runtime session using the specified options and input tensor values.
    /// </summary>
    /// <param name="session">The ONNX runtime inference session to run the inference on.</param>
    /// <param name="options">The runtime options to use during the inference.</param>
    /// <param name="ortValues">An array of input tensor values to feed into the session.</param>
    /// <returns>A task representing the asynchronous inference operation, containing the result as a disposable read-only collection of <see cref="OrtValue"/>.</returns>
    /// <exception cref="Exception">Thrown if an error occurs during inference.</exception>
    public static Task<IDisposableReadOnlyCollection<OrtValue>> RunSessionInferenceAsync(
        InferenceSession session,
        RunOptions options,
        OrtValue[] ortValues)
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