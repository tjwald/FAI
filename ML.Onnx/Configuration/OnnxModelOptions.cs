using Microsoft.ML.OnnxRuntime;
using ML.Infra.Configurations.ModelExecutors;

namespace ML.Onnx.Configuration;

public record OnnxModelExecutorOptions(
    RunOptions? RunOptions = null,
    ExecutionMode ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
    bool UseGpu = true,
    int? MaxThreads = null) : IModelExecutorConfig;