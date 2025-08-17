namespace FAI.Core.Abstractions;

/// <summary>
/// Represents a pipeline that processes input of type <typeparamref name="TInput"/>
/// and produces output of type <typeparamref name="TOutput"/>.
/// </summary>
/// <typeparam name="TInput">The type of the input data.</typeparam>
/// <typeparam name="TOutput">The type of the output data.</typeparam>
public interface IPipeline<TInput, TOutput> : IInference<TInput, TOutput>;
