using Microsoft.ML.OnnxRuntime;

namespace ML.Infra.Configurations.ModelExecutors;

public record OnnxModelExecutorOptions(
    RunOptions? RunOptions = null,
    ExecutionMode ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
    bool UseGpu = true,
    int? MaxThreads = null) : IModelExecutorConfig;