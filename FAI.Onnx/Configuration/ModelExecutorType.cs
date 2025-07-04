namespace FAI.Onnx.Configuration;

/// <summary>
/// Defines the type of model executor to use for executing models.
/// </summary>
public enum ModelExecutorType
{
    /// <summary>
    /// The default implementation of the executor. 
    /// Even though this uses async over sync while calling the onnxruntime, it does enable better parallelism.
    /// </summary>
    Simple,

    /// <summary>
    /// An executor designed to release CPU resources more effectively. 
    /// This type supports only a single thread for execution by the executor.
    /// </summary>
    Async,

    /// <summary>
    /// Similar to the <see cref="Simple"/> executor, but leverages the new API 
    /// to natively interoperate with <c>Tensor&lt;T&gt;</c>.
    /// </summary>
    Tensor
}