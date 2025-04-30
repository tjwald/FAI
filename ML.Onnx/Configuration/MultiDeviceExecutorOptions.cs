using ML.Infra.Configurations.ModelExecutors;

namespace ML.Onnx.Configuration;

public class MultiDeviceExecutorOptions: IModelExecutorConfig
{
    private readonly List<Action<OnnxModelExecutorOptions>> _configureOptions = [];

    public MultiDeviceExecutorOptions AddOptions(Action<OnnxModelExecutorOptions> configureOptions)
    {
        _configureOptions.Add(configureOptions);
        return this;
    }

    public List<OnnxModelExecutorOptions> ExecutorOptions
    {
        get
        {
            List<OnnxModelExecutorOptions> options = [];
            foreach (Action<OnnxModelExecutorOptions> configureOptions in _configureOptions)
            {
                var onnxModelExecutorOptions = new OnnxModelExecutorOptions();
                configureOptions(onnxModelExecutorOptions);
                options.Add(onnxModelExecutorOptions);
            }
            return options;
        }
    }
}