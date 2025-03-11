using System.Numerics.Tensors;
using Microsoft.ML.OnnxRuntime;
using ML.Infra.Utilities;
using ML.Onnx.Configuration;

namespace ML.Onnx.ModelExecutors;

public class OnnxModelTensorExecutor: OnnxModelExecutorBase, IOnnxModelExecutor<OnnxModelTensorExecutor>
{
    public OnnxModelTensorExecutor(InferenceSession session, RunOptions runOptions, int? maxThreadCount) : base(session, runOptions, maxThreadCount) { }

    public static OnnxModelTensorExecutor Create(InferenceSession session, RunOptions runOptions, OnnxModelExecutorOptions options)
    {
        return new OnnxModelTensorExecutor(session, runOptions, options.MaxThreads);
    }
    
    public static async Task<OnnxModelTensorExecutor> FromPretrained(string modelDir, OnnxModelExecutorOptions options)
    {
        var factory = new InferenceSessionFactory(modelDir, options);

        var session = await Task.Run(() => factory.Create());

        return Create(session, factory.RunOptions, options);
    }

    protected override OrtValue[] GetModelInputs(Tensor<long>[] inputs)
    {
        var ortValues = new OrtValue[inputs.Length];
        for (int i = 0; i < ortValues.Length; i++)
        {
            ortValues[i] = OrtValue.CreateTensorValueFromSystemNumericsTensorObject(inputs[i]);
        }
        return ortValues;
    }
    
    protected override Task<IDisposableReadOnlyCollection<OrtValue>> RunSessionInference(Tensor<long>[] inputs, OrtValue[] ortValues)
    {
        return OnnxInferenceUtils.RunSessionInferenceAsync(Session, RunOptions, ortValues);
    }
}