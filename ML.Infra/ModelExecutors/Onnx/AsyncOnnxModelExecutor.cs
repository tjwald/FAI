using System.Numerics.Tensors;
using Microsoft.ML.OnnxRuntime;
using ML.Infra.Abstractions;
using Tensor = System.Numerics.Tensors.Tensor;

namespace ML.Infra.ModelExecutors.Onnx;

public sealed class AsyncOnnxModelExecutor : OnnxModelExecutorBase, IOnnxModelExecutor<AsyncOnnxModelExecutor>
{
    private readonly SemaphoreSlim _semaphore;

    public AsyncOnnxModelExecutor(InferenceSession session) : this(session, new RunOptions())
    {
    }

    public AsyncOnnxModelExecutor(InferenceSession session, RunOptions runOptions): base(session, runOptions)
    {
        _semaphore = new SemaphoreSlim(1, 1);
    }

    public override async Task<Tensor<float>[]> RunAsync(Tensor<long>[] inputs)
    {
        await _semaphore.WaitAsync();
        try
        {
            OrtValue[] ortValues = GetModelInputs(inputs);
            var metadata = Session.OutputMetadata[Session.OutputNames[0]];
            IReadOnlyCollection<OrtValue> outputs =
                [OrtValue.CreateAllocatedTensorValue(OrtAllocator.DefaultInstance, metadata.ElementDataType, [inputs[0].Lengths[0], metadata.Dimensions[1]])];

            IReadOnlyCollection<OrtValue> result =
                await Session.RunAsync(RunOptions, Session.InputNames, ortValues, Session.OutputNames, outputs).ConfigureAwait(false);
            foreach (var input in ortValues)
            {
                input.Dispose();
            }

            return ToOutTensors(result);
        }
        finally
        {
            _semaphore.Release();
        }
    }

    public static async Task<AsyncOnnxModelExecutor> FromPretrained(string modelDir, OnnxModelExecutorOptions options)
    {
        var factory = new InferenceSessionFactory(modelDir, options);

        var session = await Task.Run(() => factory.Create());

        return Create(session, factory.RunOptions, options);
    }
    
    public static AsyncOnnxModelExecutor Create(InferenceSession session, RunOptions runOptions, OnnxModelExecutorOptions options)
    {
        return new AsyncOnnxModelExecutor(session, runOptions);
    }
}