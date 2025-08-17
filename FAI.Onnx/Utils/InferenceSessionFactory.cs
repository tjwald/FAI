using FAI.Onnx.Configuration;
using Microsoft.ML.OnnxRuntime;

namespace FAI.Onnx.Utils;

/// <summary>
/// Factory for creating instances of <see cref="InferenceSession"/> using specified ONNX options.
/// </summary>
internal sealed class InferenceSessionFactory
{
    private readonly string _modelPath;
    private readonly SessionOptions _sessionOptions;

    /// <summary>
    /// Gets the runtime options to be used during inference.
    /// </summary>
    public RunOptions RunOptions { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="InferenceSessionFactory"/> class
    /// with the specified ONNX configuration options.
    /// </summary>
    /// <param name="options">The ONNX options used to configure the inference session.</param>
    public InferenceSessionFactory(OnnxOptions options)
    {
        _modelPath = options.FullModelPath;
        _sessionOptions = options.SessionOptions;
        RunOptions = options.RunOptions;
    }

    /// <summary>
    /// Creates a new instance of <see cref="InferenceSession"/> configured with the factory's settings.
    /// </summary>
    /// <returns>A newly created <see cref="InferenceSession"/>.</returns>
    public InferenceSession Create()
    {
        return new InferenceSession(_modelPath, _sessionOptions);
    }
}
