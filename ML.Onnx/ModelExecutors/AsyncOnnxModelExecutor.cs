using System.Collections;
using System.Numerics.Tensors;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using ML.Onnx.Configuration;
using Tensor = System.Numerics.Tensors.Tensor;

namespace ML.Onnx.ModelExecutors;

public sealed class AsyncOnnxModelExecutor : OnnxModelExecutorBase, IOnnxModelExecutor<AsyncOnnxModelExecutor>
{
    private readonly long[] _outputDimensions;
    private readonly TensorElementType _elementDataType;
    
    public AsyncOnnxModelExecutor(InferenceSession session, RunOptions runOptions) : base(session, runOptions, maxThreads: 1)
    {
        var metadata = Session.OutputMetadata[Session.OutputNames[0]];
        _elementDataType = metadata.ElementDataType;
        _outputDimensions = new long[metadata.Dimensions.Length - 1];
        Tensor.ConvertChecked(new ReadOnlyTensorSpan<int>(metadata.Dimensions.AsSpan(1)), new TensorSpan<long>(_outputDimensions));
    }

    protected override async Task<IDisposableReadOnlyCollection<OrtValue>> RunSessionInference(System.Numerics.Tensors.Tensor<long>[] inputs, OrtValue[] ortValues)
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
    
    public static async Task<AsyncOnnxModelExecutor> FromPretrained(OnnxModelExecutorOptions options)
    {
        var factory = new InferenceSessionFactory(options.OnnxOptions);

        var session = await Task.Run(() => factory.Create());

        return Create(session, factory.RunOptions, options);
    }

    public static AsyncOnnxModelExecutor Create(InferenceSession session, RunOptions runOptions, OnnxModelExecutorOptions options)
    {
        return new AsyncOnnxModelExecutor(session, runOptions);
    }
}

file struct DisposableCollection<T>: IDisposableReadOnlyCollection<T> where T : IDisposable
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
        if (_disposed || !disposing) return;
        
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