using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace ML.Infra;

/// <summary>
/// Provides extension methods for working with tensors, including methods to retrieve spans and memory representations.
/// </summary>
public static class TensorExtensions
{
    /// <summary>
    /// Retrieves a span representing a specific row of a tensor.
    /// </summary>
    /// <typeparam name="T">The type of elements in the tensor.</typeparam>
    /// <param name="tensor">The tensor span to retrieve the row from.</param>
    /// <param name="i">The index of the row to retrieve.</param>
    /// <returns>A span representing the specified row of the tensor.</returns>
    public static Span<T> GetRowSpan<T>(this TensorSpan<T> tensor, int i)
    {
        TensorSpan<T> tensorRow = tensor.Slice(i..(i + 1), ..);
        return MemoryMarshal.CreateSpan(ref tensorRow.GetPinnableReference(), (int)tensor.Lengths[1]);
    }

    /// <summary>
    /// Retrieves a read-only span representing a specific row of a read-only tensor.
    /// </summary>
    /// <typeparam name="T">The type of elements in the tensor.</typeparam>
    /// <param name="tensor">The read-only tensor span to retrieve the row from.</param>
    /// <param name="i">The index of the row to retrieve.</param>
    /// <returns>A read-only span representing the specified row of the tensor.</returns>
    public static ReadOnlySpan<T> GetRowSpan<T>(this ReadOnlyTensorSpan<T> tensor, int i)
    {
        ReadOnlyTensorSpan<T> tensorRow = tensor.Slice(i..(i + 1), ..);
        return MemoryMarshal.CreateReadOnlySpan(in tensorRow.GetPinnableReference(), (int)tensor.Lengths[1]);
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
        return ExternalClassAccessor<T>.GetValues(tensor).AsMemory();
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
