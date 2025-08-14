using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace FAI.Core;

/// <summary>
/// Provides extension methods for working with tensors, including methods to retrieve spans and memory representations.
/// </summary>
public static class TensorExtensions
{
    public static Span<T> AsSpan<T>(this TensorSpan<T> tensor)
    {
        return MemoryMarshal.CreateSpan(ref TensorMarshal.GetReference(tensor), (int)tensor.FlattenedLength);
    }

    public static ReadOnlySpan<T> AsSpan<T>(this ReadOnlyTensorSpan<T> tensor)
    {
        return MemoryMarshal.CreateReadOnlySpan(in TensorMarshal.GetReference(tensor), (int)tensor.FlattenedLength);
    }

    /// <summary>
    /// Converts a tensor to a memory representation.
    /// </summary>
    /// <typeparam name="T">The type of elements in the tensor.</typeparam>
    /// <param name="tensor">The tensor to convert to memory.</param>
    /// <returns>A memory representation of the tensor.</returns>
    /// <remarks>
    /// This method relies on an external accessor to retrieve the underlying values of the tensor.
    /// </remarks>
    public static Memory<T> AsMemory<T>(this Tensor<T> tensor)
    {
        // Would like this code to be safe!
        return ExternalClassAccessor<T>.GetValues(tensor).AsMemory(0, (int)tensor.FlattenedLength);
    }
}

/// <summary>
/// Provides access to internal fields of the Tensor class.
/// </summary>
/// <typeparam name="T">The type of elements in the tensor.</typeparam>
internal static class ExternalClassAccessor<T>
{
    /// <summary>
    /// Retrieves a reference to the internal values array of a tensor.
    /// </summary>
    /// <param name="instance">The tensor instance to access.</param>
    /// <returns>A reference to the internal values array of the tensor.</returns>
    [UnsafeAccessor(UnsafeAccessorKind.Field, Name = "_values")]
    public static extern ref T[] GetValues(Tensor<T> instance);
}