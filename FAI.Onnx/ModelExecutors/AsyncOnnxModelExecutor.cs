using System.Collections;
using System.Numerics.Tensors;
using FAI.Onnx.Configuration;
using FAI.Onnx.Utils;
using Microsoft.ML.OnnxRuntime;
using TensorElementType = Microsoft.ML.OnnxRuntime.Tensors.TensorElementType;

namespace FAI.Onnx.ModelExecutors;

/// <summary>
/// Represents an ONNX model executor designed to release CPU resources more effectively.
/// This executor supports asynchronous operations but limits execution to a single thread.
/// </summary>
public sealed class AsyncOnnxModelExecutor : OnnxModelExecutorBase, IOnnxModelExecutor<AsyncOnnxModelExecutor>
{
    private readonly long[] _outputDimensions;
    private readonly TensorElementType _elementDataType;

    /// <summary>
    /// Initializes a new instance of the <see cref="AsyncOnnxModelExecutor"/> class.
    /// </summary>
    /// <param name="session">The ONNX runtime inference session to use.</param>
    /// <param name="runOptions">The runtime options for execution.</param>
    public AsyncOnnxModelExecutor(InferenceSession session, RunOptions runOptions)
        : base(session, runOptions, maxThreads: 1)
    {
        var metadata = Session.OutputMetadata[Session.OutputNames[0]];
        _elementDataType = metadata.ElementDataType;
        _outputDimensions = new long[metadata.Dimensions.Length - 1];
        Tensor.ConvertChecked(new ReadOnlyTensorSpan<int>(metadata.Dimensions.AsSpan(1)), new TensorSpan<long>(_outputDimensions));
    }

    /// <summary>
    /// Performs asynchronous inference using the ONNX runtime session.
    /// </summary>
    /// <param name="inputs">The input tensors for the model.</param>
    /// <param name="ortValues">The prepared ONNX tensor values.</param>
    /// <returns>
    /// A task representing the asynchronous inference operation, containing the result
    /// as a disposable collection of <see cref="OrtValue"/>.
    /// </returns>
    protected override async Task<IDisposableReadOnlyCollection<OrtValue>> RunSessionInference(Tensor<long>[] inputs, OrtValue[] ortValues)
    {
        long[] outputDimensions = new long[_outputDimensions.Length + 1];
        outputDimensions[0] = inputs[0].Lengths[0];
        _outputDimensions.AsSpan().CopyTo(outputDimensions.AsSpan(1));

        IReadOnlyCollection<OrtValue> outputs =
            [OrtValue.CreateAllocatedTensorValue(OrtAllocator.DefaultInstance, _elementDataType, outputDimensions)];

        IReadOnlyCollection<OrtValue> result =
            await Session.RunAsync(RunOptions, Session.InputNames, ortValues, Session.OutputNames, outputs).ConfigureAwait(false);

        return new DisposableCollection<OrtValue>(result);
    }

    /// <summary>
    /// Creates an instance of the <see cref="AsyncOnnxModelExecutor"/> asynchronously using pre-trained model options.
    /// </summary>
    /// <param name="options">The configuration options for the model executor.</param>
    /// <returns>A task representing the asynchronous operation, containing the created <see cref="AsyncOnnxModelExecutor"/>.</returns>
    public static AsyncOnnxModelExecutor FromPretrained(OnnxModelExecutorOptions options)
    {
        var factory = new InferenceSessionFactory(options.OnnxOptions);

        var session = factory.Create();

        return Create(session, factory.RunOptions, options);
    }

    /// <summary>
    /// Creates an instance of the <see cref="AsyncOnnxModelExecutor"/> using the specified session, options, and configuration.
    /// </summary>
    /// <param name="session">The ONNX runtime inference session to use.</param>
    /// <param name="runOptions">The runtime options for execution.</param>
    /// <param name="options">The configuration options for the model executor.</param>
    /// <returns>A new instance of <see cref="AsyncOnnxModelExecutor"/>.</returns>
    public static AsyncOnnxModelExecutor Create(InferenceSession session, RunOptions runOptions, OnnxModelExecutorOptions options)
    {
        return new AsyncOnnxModelExecutor(session, runOptions);
    }
}

file struct DisposableCollection<T> : IDisposableReadOnlyCollection<T> where T : IDisposable
{
    private bool _disposed;
    private readonly T[] _collection;

    public DisposableCollection(IReadOnlyCollection<T> collection)
    {
        _collection = collection.ToArray();
    }

    #region IDisposable Support

    private void Dispose(bool disposing)
    {
        if (_disposed || !disposing)
            return;

        // Dispose in the reverse order.
        // Objects should typically be destroyed/disposed
        // in the reverse order of its creation
        // especially if the objects created later refer to the
        // objects created earlier. For homogeneous collections of objects
        // it would not matter.
        for (int i = _collection.Length - 1; i >= 0; i--)
        {
            var item = _collection[i];
            item.Dispose();
        }

        _disposed = true;
    }

    public void Dispose()
    {
        Dispose(true);
    }

    #endregion

    public IEnumerator<T> GetEnumerator()
    {
        return _collection.AsEnumerable().GetEnumerator();
    }

    IEnumerator IEnumerable.GetEnumerator()
    {
        return GetEnumerator();
    }

    public int Count => _collection.Length;

    public T this[int index] => _collection[index];
}
