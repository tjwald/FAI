namespace ML.Onnx.Utils;

/// <summary>
/// A thread-safe circular counter that increments atomically and wraps around when it reaches a specified maximum value.
/// </summary>
public sealed class CircularAtomicCounter
{
    private readonly int _maxValue;
    private uint _currentValue;

    /// <summary>
    /// Initializes a new instance of the <see cref="CircularAtomicCounter"/> class.
    /// </summary>
    /// <param name="maxValue">The maximum value the counter can reach before wrapping around.</param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="maxValue"/> is less than or equal to zero.</exception>
    public CircularAtomicCounter(int maxValue)
    {
        if (maxValue <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(maxValue), "Max value must be greater than zero.");
        }

        _maxValue = maxValue;
        _currentValue = uint.MaxValue;
    }

    /// <summary>
    /// Gets the next value in the circular sequence.
    /// </summary>
    /// <returns>The next value in the sequence, wrapping around to zero after reaching the maximum value.</returns>
    public int Next()
    {
        return (int)(Interlocked.Increment(ref _currentValue) % _maxValue);
    }
}