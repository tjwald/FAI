using System.Numerics.Tensors;
using Microsoft.ML.OnnxRuntime;
using ML.Onnx.Configuration;

namespace ML.Onnx.ModelExecutors;

public sealed class OnnxModelExecutor : OnnxModelExecutorBase, IOnnxModelExecutor<OnnxModelExecutor>
{
    public OnnxModelExecutor(InferenceSession session, RunOptions runOptions, int? maxThreads = null) : base(session, runOptions, maxThreads)
    {
    }

    protected override Task<IDisposableReadOnlyCollection<OrtValue>> RunSessionInference(Tensor<long>[] inputs, OrtValue[] ortValues)
    {
        return OnnxInferenceUtils.RunSessionInferenceAsync(Session, RunOptions, ortValues);
    }

    public static async Task<OnnxModelExecutor> FromPretrained(OnnxModelExecutorOptions options)
    {
        var factory = new InferenceSessionFactory(options.OnnxOptions);

        var session = await Task.Run(() => factory.Create());

        return Create(session, factory.RunOptions, options);
    }

    public static OnnxModelExecutor Create(InferenceSession session, RunOptions runOptions, OnnxModelExecutorOptions options)
    {
        return new OnnxModelExecutor(session, runOptions, options.MaxThreads);
    }
}