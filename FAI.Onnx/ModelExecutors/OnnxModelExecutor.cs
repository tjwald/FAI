using System.Numerics.Tensors;
using FAI.Onnx.Configuration;
using FAI.Onnx.Utils;
using Microsoft.ML.OnnxRuntime;

namespace FAI.Onnx.ModelExecutors;

/// <summary>
/// Onnxruntime model executor. Uses async over sync while calling the onnxruntime, it does so to enable better parallelism.
/// </summary>
public sealed class OnnxModelExecutor : OnnxModelExecutorBase, IOnnxModelExecutor<OnnxModelExecutor>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="OnnxModelExecutor"/> class.
    /// </summary>
    /// <param name="session">The ONNX runtime inference session to use.</param>
    /// <param name="runOptions">The runtime options for execution.</param>
    /// <param name="maxThreads">The maximum number of threads allowed, or <c>null</c> for no limit.</param>
    public OnnxModelExecutor(InferenceSession session, RunOptions runOptions, int? maxThreads = null) : base(session, runOptions, maxThreads)
    {
    }

    /// <summary>
    /// Performs inference using the ONNX runtime session asynchronously.
    /// </summary>
    /// <param name="inputs">The input tensors for the model.</param>
    /// <param name="ortValues">The prepared ONNX tensor values.</param>
    /// <returns>A task representing the asynchronous inference operation, containing the result as a disposable collection of <see cref="OrtValue"/>.</returns>
    protected override Task<IDisposableReadOnlyCollection<OrtValue>> RunSessionInference(Tensor<long>[] inputs, OrtValue[] ortValues)
    {
        return OnnxInferenceUtils.RunSessionInferenceAsync(Session, RunOptions, ortValues);
    }

    /// <summary>
    /// Creates a new instance of the <see cref="OnnxModelExecutor"/> from pre-trained model options asynchronously.
    /// </summary>
    /// <param name="options">The configuration options for the model executor.</param>
    /// <returns>A task representing the asynchronous operation, containing the created <see cref="OnnxModelExecutor"/>.</returns>
    public static async Task<OnnxModelExecutor> FromPretrained(OnnxModelExecutorOptions options)
    {
        var factory = new InferenceSessionFactory(options.OnnxOptions);
        var session = await Task.Run(() => factory.Create());
        return Create(session, factory.RunOptions, options);
    }

    /// <summary>
    /// Creates a new instance of the <see cref="OnnxModelExecutor"/> with the specified session, options, and configuration.
    /// </summary>
    /// <param name="session">The ONNX runtime inference session to use.</param>
    /// <param name="runOptions">The runtime options for execution.</param>
    /// <param name="options">The configuration options for the model executor.</param>
    /// <returns>A new instance of <see cref="OnnxModelExecutor"/>.</returns>
    public static OnnxModelExecutor Create(InferenceSession session, RunOptions runOptions, OnnxModelExecutorOptions options)
    {
        return new OnnxModelExecutor(session, runOptions, options.MaxThreads);
    }
}
