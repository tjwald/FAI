using Microsoft.ML.OnnxRuntime;

namespace ML.Infra;


public record OnnxModelRunerOptions(
    RunOptions? RunOptions = null,
    ExecutionMode ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
    bool UseGpu = true);


public class ModelRunner
{
    private readonly InferenceSession _session;
    private readonly RunOptions _runOptions;

    public ModelRunner(InferenceSession session) : this(session, new RunOptions()) { }

    public ModelRunner(InferenceSession session, RunOptions runOptions)
    {
        _session = session;
        _runOptions = runOptions;
    }


    public Task<IDisposableReadOnlyCollection<DisposableNamedOnnxValue>> RunAsync(NamedOnnxValue[] inputs)
    {
        return Task.Run(() => _session.Run(inputs, _session.OutputNames, _runOptions));
    }

    public async Task<IDisposableReadOnlyCollection<OrtValue>> RunAsync(Memory<long>[] inputs, long[] dims)
    {
        var inputsOrts = new OrtValue[inputs.Length];
        for (int i = 0; i < inputs.Length; i++)
        {
            inputsOrts[i] = OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, inputs[i], dims);
        }

        return await RunAsync(inputsOrts);
    }

    public async Task<IDisposableReadOnlyCollection<OrtValue>> RunAsync(OrtValue[] inputs)
    {
        var result = await Task.Run(() => _session.Run(_runOptions, _session.InputNames, inputs, _session.OutputNames)).ConfigureAwait(false);

        for (int i = 0; i < inputs.Length; i++)
        {
            inputs[i].Dispose();
        }
        return result;
    }

    public static async Task<ModelRunner> FromPretrained(string modelDir, OnnxModelRunerOptions options)
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

        return new ModelRunner(session, options.RunOptions ?? new RunOptions());
    }
}
