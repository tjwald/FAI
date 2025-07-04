using System.Numerics.Tensors;
using FAI.Onnx.Configuration;
using FAI.Onnx.Utils;
using Microsoft.ML.OnnxRuntime;

namespace FAI.Onnx.ModelExecutors;

/// <summary>
/// Represents an ONNX model executor that uses the new API to natively interoperate with <c>Tensor&lt;T&gt;</c>.
/// </summary>
public sealed class OnnxModelTensorExecutor : OnnxModelExecutorBase, IOnnxModelExecutor<OnnxModelTensorExecutor>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="OnnxModelTensorExecutor"/> class.
    /// </summary>
    /// <param name="session">The ONNX runtime inference session to use.</param>
    /// <param name="runOptions">The runtime options for execution.</param>
    /// <param name="maxThreadCount">The maximum number of threads allowed, or <c>null</c> for no limit.</param>
    public OnnxModelTensorExecutor(InferenceSession session, RunOptions runOptions, int? maxThreadCount)
        : base(session, runOptions, maxThreadCount)
    {
    }

    /// <summary>
    /// Creates an instance of the <see cref="OnnxModelTensorExecutor"/> using the specified session, options, and configuration.
    /// </summary>
    /// <param name="session">The ONNX runtime inference session to use.</param>
    /// <param name="runOptions">The runtime options for execution.</param>
    /// <param name="options">The configuration options for the model executor.</param>
    /// <returns>A new instance of <see cref="OnnxModelTensorExecutor"/>.</returns>
    public static OnnxModelTensorExecutor Create(InferenceSession session, RunOptions runOptions, OnnxModelExecutorOptions options)
    {
        return new OnnxModelTensorExecutor(session, runOptions, options.MaxThreads);
    }

    /// <summary>
    /// Creates an instance of the <see cref="OnnxModelTensorExecutor"/> asynchronously using pre-trained model options.
    /// </summary>
    /// <param name="options">The configuration options for the model executor.</param>
    /// <returns>A task representing the asynchronous operation, containing the created <see cref="OnnxModelTensorExecutor"/>.</returns>
    public static async Task<OnnxModelTensorExecutor> FromPretrained(OnnxModelExecutorOptions options)
    {
        var factory = new InferenceSessionFactory(options.OnnxOptions);

        var session = await Task.Run(() => factory.Create());

        return Create(session, factory.RunOptions, options);
    }

    /// <summary>
    /// Prepares input tensors as ONNX runtime tensor values, leveraging native interop with <c>Tensor&lt;T&gt;</c>.
    /// </summary>
    /// <param name="inputs">The input tensors for the model.</param>
    /// <returns>An array of prepared <see cref="OrtValue"/> tensors.</returns>
    protected override OrtValue[] GetModelInputs(Tensor<long>[] inputs)
    {
        var ortValues = new OrtValue[inputs.Length];
        for (int i = 0; i < ortValues.Length; i++)
        {
            ortValues[i] = OrtValue.CreateTensorValueFromSystemNumericsTensorObject(inputs[i]);
        }

        return ortValues;
    }

    /// <summary>
    /// Performs inference using the ONNX runtime session asynchronously.
    /// </summary>
    /// <param name="inputs">The input tensors for the model.</param>
    /// <param name="ortValues">The prepared ONNX tensor values.</param>
    /// <returns>
    /// A task representing the asynchronous inference operation, containing the result
    /// as a disposable collection of <see cref="OrtValue"/>.
    /// </returns>
    protected override Task<IDisposableReadOnlyCollection<OrtValue>> RunSessionInference(Tensor<long>[] inputs, OrtValue[] ortValues)
    {
        return OnnxInferenceUtils.RunSessionInferenceAsync(Session, RunOptions, ortValues);
    }
}