using System.Numerics.Tensors;
using Microsoft.ML.OnnxRuntime;
using ML.Infra.Utilities;
using ML.Onnx.Configuration;

namespace ML.Onnx.ModelExecutors;

public sealed class OnnxModelExecutor : OnnxModelExecutorBase, IOnnxModelExecutor<OnnxModelExecutor>
{
    private readonly SemaphoreSlim? _semaphore;
    
    public OnnxModelExecutor(InferenceSession session) : this(session, new RunOptions())
    {
    }

    public OnnxModelExecutor(InferenceSession session, RunOptions runOptions, int? maxThreads = null) : base(session, runOptions)
    {
        _semaphore = maxThreads.HasValue ? new SemaphoreSlim(maxThreads.Value, maxThreads.Value) : null;
    }

    private Task<IDisposableReadOnlyCollection<OrtValue>> RunWithThreadPool(OrtValue[] ortValues)
    {
        var tcs = new TaskCompletionSource<IDisposableReadOnlyCollection<OrtValue>>();
        ThreadPool.QueueUserWorkItem(ortValues =>
        {
            try
            {
                IDisposableReadOnlyCollection<OrtValue> x = Session.Run(RunOptions, Session.InputNames, ortValues, Session.OutputNames);
                tcs.SetResult(x);
            }
            catch (Exception e)
            {
                tcs.SetException(e);
            }
        }, ortValues, true);
        return tcs.Task;
    }

    public override async Task<Tensor<float>[]> RunAsync(Tensor<long>[] inputs)
    {
        OrtValue[] ortValues = GetModelInputs(inputs);

        IDisposableReadOnlyCollection<OrtValue> result;
        using (await _semaphore.EnterScope())
        {
            result = await RunWithThreadPool(ortValues);
        }

        foreach (var input in ortValues)
        {
            input.Dispose();
        }

        Tensor<float>[] outTensors = ToOutTensors(result);

        return outTensors;
    }

    public static async Task<OnnxModelExecutor> FromPretrained(string modelDir, OnnxModelExecutorOptions options)
    {
        var factory = new InferenceSessionFactory(modelDir, options);

        var session = await Task.Run(() => factory.Create());

        return Create(session, factory.RunOptions, options);
    }
    
    public static OnnxModelExecutor Create(InferenceSession session, RunOptions runOptions, OnnxModelExecutorOptions options)
    {
        return new OnnxModelExecutor(session, runOptions, options.MaxThreads);
    }
}