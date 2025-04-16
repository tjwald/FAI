using Microsoft.ML.OnnxRuntime;
using ML.Infra.Configurations.ModelExecutors;

namespace ML.Onnx.Configuration;

public class OnnxModelExecutorOptions : IModelExecutorConfig
{
    private Action<OnnxOptions>? _configureOptions;
    
    public OnnxOptions OnnxOptions
    {
        get
        {
            var onnxOptions = new OnnxOptions();
            _configureOptions?.Invoke(onnxOptions);
            return onnxOptions;
        }
    }

    public int? MaxThreads { get; set; } = null;

    public OnnxModelExecutorOptions ConfigureOnnxOptions(Action<OnnxOptions> configureOptions)
    {
        _configureOptions = configureOptions;
        return this;
    }
}


public class OnnxOptions
{
    /// <summary>
    /// The name of the model file to load. 
    /// Default is "model_optimized.onnx".
    /// </summary>
    public string ModelFileName { get; set; } = "model_optimized.onnx";

    /// <summary>
    /// The directory where the model file is located. 
    /// Default is the current directory ("./").
    /// </summary>
    public string ModelDir { get; set; } = ".";

    /// <summary>
    /// The full path of the model file, lazily computed by combining 
    /// <see cref="ModelDir"/> and <see cref="ModelFileName"/>.
    /// </summary>
    public string FullModelPath => Path.Combine(ModelDir, ModelFileName);

    /// <summary>
    /// Internal handler for configuring <see cref="Microsoft.ML.OnnxRuntime.SessionOptions"/>.
    /// </summary>
    private Action<SessionOptions>? CreateSessionOptionsDelegate { get; set; }

    /// <summary>
    /// Internal handler for configuring <see cref="Microsoft.ML.OnnxRuntime.RunOptions"/>.
    /// </summary>
    private Action<RunOptions>? CreateRunOptionsDelegate { get; set; }

    /// <summary>
    /// Configures <see cref="Microsoft.ML.OnnxRuntime.SessionOptions"/> using the provided 
    /// configuration function. The configuration function replaces any previously set function.
    /// </summary>
    /// <param name="configurator">
    /// A function that accepts the default <see cref="Microsoft.ML.OnnxRuntime.SessionOptions"/> 
    /// instance and returns a modified instance.
    /// </param>
    /// <returns>
    /// The current <see cref="OnnxModelExecutorOptions"/> instance to allow chaining.
    /// </returns>
    public OnnxOptions ConfigureSessionOptions(Action<SessionOptions> configurator)
    {
        CreateSessionOptionsDelegate = configurator;
        return this;
    }

    /// <summary>
    /// Configures <see cref="Microsoft.ML.OnnxRuntime.RunOptions"/> using the provided 
    /// configuration function. The configuration function replaces any previously set function.
    /// </summary>
    /// <param name="configurator">
    /// A function that accepts the default <see cref="Microsoft.ML.OnnxRuntime.RunOptions"/> 
    /// instance and returns a modified instance.
    /// </param>
    /// <returns>
    /// The current <see cref="OnnxModelExecutorOptions"/> instance to allow chaining.
    /// </returns>
    public OnnxOptions ConfigureRunOptions(Action<RunOptions> configurator)
    {
        CreateRunOptionsDelegate = configurator;
        return this;
    }

    /// <summary>
    /// Returns a configured instance of <see cref="SessionOptions"/>.
    /// If no configuration function was provided, returns a default instance.
    /// </summary>
    /// <value>The configured <see cref="SessionOptions"/> instance.</value>
    public SessionOptions SessionOptions
    {
        get
        {
            var sessionOptions = new SessionOptions();
            CreateSessionOptionsDelegate?.Invoke(sessionOptions);
            return sessionOptions;
        }
    }

    /// <summary>
    /// Returns a configured instance of <see cref="RunOptions"/>.
    /// If no configuration function was provided, returns a default instance.
    /// </summary>
    /// <value>The configured <see cref="RunOptions"/> instance.</value>
    public RunOptions RunOptions
    {
        get
        {
            var runOptions = new RunOptions();
            CreateRunOptionsDelegate?.Invoke(runOptions);
            return runOptions;
        }
    }
}