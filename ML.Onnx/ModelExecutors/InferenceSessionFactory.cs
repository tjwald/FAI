using Microsoft.ML.OnnxRuntime;
using ML.Onnx.Configuration;

namespace ML.Onnx.ModelExecutors;

public sealed class InferenceSessionFactory
{
    private readonly string _modelPath;
    private readonly SessionOptions _sessionOptions;

    public RunOptions RunOptions { get; }

    public InferenceSessionFactory(OnnxOptions options)
    {
        _modelPath = options.FullModelPath;

        _sessionOptions = options.SessionOptions;
        RunOptions = options.RunOptions;
    }

    public InferenceSession Create()
    {
        return new InferenceSession(_modelPath, _sessionOptions);
    }
}