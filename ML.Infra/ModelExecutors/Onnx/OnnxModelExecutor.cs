using System.Numerics.Tensors;
using Microsoft.ML.OnnxRuntime;

namespace ML.Infra.ModelExecutors.Onnx;

public record OnnxModelExecutorOptions(
    RunOptions? RunOptions = null,
    ExecutionMode ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
    bool UseGpu = true,
    int MaxInferenceSessions = 1,
    int? MaxThreads = null);

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

    public override async Task<Tensor<float>[]> RunAsync(Tensor<long>[] inputs)
    {
        OrtValue[] ortValues = GetModelInputs(inputs);

        if (_semaphore is not null)
        {
            await _semaphore.WaitAsync();
        }
        
        IDisposableReadOnlyCollection<OrtValue> result = await Task.Run(
            () => Session.Run(RunOptions, Session.InputNames, ortValues, Session.OutputNames)).ConfigureAwait(false);

        _semaphore?.Release();

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