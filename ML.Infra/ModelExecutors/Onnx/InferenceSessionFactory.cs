using Microsoft.ML.OnnxRuntime;

namespace ML.Infra.ModelExecutors.Onnx;

public class InferenceSessionFactory
{
    private readonly string _modelDir;
    private readonly SessionOptions _sessionOptions;
    public RunOptions RunOptions { get; }

    public InferenceSessionFactory(string modelDir, OnnxModelExecutorOptions options)
    {
        _modelDir = modelDir;
        RunOptions = options.RunOptions ?? new RunOptions();
        _sessionOptions = new SessionOptions();
        
        if (options.UseGpu)
        {
            _sessionOptions.AppendExecutionProvider_CUDA();
            Console.WriteLine("Using GPU accelerator");
        }

        _sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_EXTENDED;
        _sessionOptions.AddSessionConfigEntry("session.use_fp16", "1");
        _sessionOptions.AppendExecutionProvider_CPU();

        _sessionOptions.ExecutionMode = options.ExecutionMode;
        Console.WriteLine($"Running With Execution Mode: {options.ExecutionMode}");
    }

    public InferenceSession Create()
    {
        return new InferenceSession(Path.Combine(_modelDir, "model_optimized.onnx"), _sessionOptions);
    }
}