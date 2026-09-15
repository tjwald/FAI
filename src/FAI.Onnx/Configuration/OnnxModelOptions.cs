using FAI.Core.Configurations.ModelExecutors;
using Microsoft.ML.OnnxRuntime;

namespace FAI.Onnx.Configuration;

/// <summary>
/// Represents the configuration options for the OnnxModelExecutor.
/// </summary>
public class OnnxModelExecutorOptions : IModelExecutorOptions
{
    private Action<OnnxOptions>? _configureOptions;

    public ModelExecutorType ModelExecutorType { get; set; } = ModelExecutorType.Simple;

    /// <summary>
    /// Gets the OnnxOptions object, configured using the provided delegate if available.
    /// </summary>
    public OnnxOptions OnnxOptions
    {
        get
        {
            var onnxOptions = new OnnxOptions();
            _configureOptions?.Invoke(onnxOptions);
            return onnxOptions;
        }
    }

    /// <summary>
    /// Gets or sets the maximum number of threads that can be used for execution.
    /// </summary>
    public int? MaxThreads { get; set; } = null;

    /// <summary>
    /// Configures the OnnxOptions using the specified delegate.
    /// </summary>
    /// <param name="configureOptions">A delegate to configure the OnnxOptions.</param>
    /// <returns>The current instance of <see cref="OnnxModelExecutorOptions"/>.</returns>
    public OnnxModelExecutorOptions ConfigureOnnxOptions(Action<OnnxOptions> configureOptions)
    {
        _configureOptions = configureOptions;
        return this;
    }
}

/// <summary>
/// Provides configuration options for ONNX (Open Neural Network Exchange) model execution.
/// </summary>
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

    private SessionOptions? _sessionOptions;

    /// <summary>
    /// Returns a configured instance of <see cref="SessionOptions"/>.
    /// If no configuration function was provided, returns a default instance.
    /// </summary>
    /// <value>The configured <see cref="SessionOptions"/> instance.</value>
    public SessionOptions SessionOptions
    {
        get
        {
            if (_sessionOptions is not null)
                return _sessionOptions;

            var sessionOptions = new SessionOptions();
            CreateSessionOptionsDelegate?.Invoke(sessionOptions);

            _sessionOptions = sessionOptions;
            return sessionOptions;
        }
    }

    private RunOptions? _runOptions;

    /// <summary>
    /// Returns a configured instance of <see cref="RunOptions"/>.
    /// If no configuration function was provided, returns a default instance.
    /// </summary>
    /// <value>The configured <see cref="RunOptions"/> instance.</value>
    public RunOptions RunOptions
    {
        get
        {
            if (_runOptions is not null)
                return _runOptions;

            var runOptions = new RunOptions();
            CreateRunOptionsDelegate?.Invoke(runOptions);

            _runOptions = runOptions;
            return runOptions;
        }
    }
}
