using System.Numerics.Tensors;
using Microsoft.ML.OnnxRuntime;
using ML.Infra.Abstractions;

namespace ML.Infra.ModelExecutors.Onnx;


public record OnnxModelExecutorOptions(
    RunOptions? RunOptions = null,
    ExecutionMode ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
    bool UseGpu = true);


public class OnnxModelExecutor: IModelExecutor<long, float>
{
    private readonly InferenceSession _session;
    private readonly RunOptions _runOptions;

    public OnnxModelExecutor(InferenceSession session) : this(session, new RunOptions()) { }

    public OnnxModelExecutor(InferenceSession session, RunOptions runOptions)
    {
        _session = session;
        _runOptions = runOptions;
    }

    public async Task<Tensor<float>[]> RunAsync(Tensor<long>[] inputs)
    {
        OrtValue[] ortValues = GetModelInputs(inputs);
        
        IDisposableReadOnlyCollection<OrtValue> result = await Task.Run(
            () => _session.Run(_runOptions, _session.InputNames, ortValues, _session.OutputNames)).ConfigureAwait(false);
        
        foreach (var input in ortValues)
        {
            input.Dispose();
        }
        
        Tensor<float>[] outTensors = ToOutTensors(result);

        return outTensors;
    }

    private static OrtValue[] GetModelInputs(Tensor<long>[] inputs)
    {
        long[] dims = GetInputDims(inputs);

        Span<Memory<long>> modelInputs = GetInputsAsMemory(inputs);

        OrtValue[] ortValues = modelInputs.ToOrtValues(dims);
        return ortValues;
    }

    private static Tensor<float>[] ToOutTensors(IDisposableReadOnlyCollection<OrtValue> result)
    {
        var outTensors = new Tensor<float>[result.Count];
        for (int i = 0; i < outTensors.Length; i++)
        {
            long[] outDims = result[i].GetTensorTypeAndShape().Shape!;
            nint[] outDimsAsNInts = new nint[outDims.Length];
            for (int dim = 0; dim < outDims.Length; dim++)
            {
                outDimsAsNInts[dim] = (nint)outDims[dim];
            }

            Tensor<float> outTensor = Tensor.Create<float>(outDimsAsNInts);
            result[i].GetTensorDataAsSpan<float>().CopyTo(outTensor.AsMemory().Span);
            outTensors[i] = outTensor;
        }
        result.Dispose();
        return outTensors;
    }

    private static Span<Memory<long>> GetInputsAsMemory(Tensor<long>[] inputs)
    {
        Span<Memory<long>> modelInputs = new Memory<long>[inputs.Length];
        for (int i = 0; i < modelInputs.Length; i++)
        {
            modelInputs[i] = inputs[i].AsMemory();
        }

        return modelInputs;
    }

    private static long[] GetInputDims(Tensor<long>[] inputs)
    {
        long[] dims = new long[inputs[0].Rank];
        for (int i = 0; i < inputs[0].Rank; i++)
        {
            dims[i] = inputs[0].Lengths[i];
        }

        return dims;
    }

    public static async Task<OnnxModelExecutor> FromPretrained(string modelDir, OnnxModelExecutorOptions options)
    {
        SessionOptions sessionOptions = new();
        if (options.UseGpu)
        {
            sessionOptions.AppendExecutionProvider_CUDA();
            Console.WriteLine("Using GPU accelerator");
        }

        sessionOptions.AppendExecutionProvider_CPU();

        sessionOptions.ExecutionMode = options.ExecutionMode;
        Console.WriteLine($"Running With Execution Mode: {options.ExecutionMode}");

        var session = await Task.Run(() => new InferenceSession(Path.Combine(modelDir, "model.onnx"), sessionOptions));

        return new OnnxModelExecutor(session, options.RunOptions ?? new RunOptions());
    }
}
