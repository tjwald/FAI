namespace ML.Infra.Utilities;

public class CircularAtomicCounter
{
    private readonly int _maxValue;
    private uint _currentValue;

    public CircularAtomicCounter(int maxValue)
    {
        if (maxValue <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(maxValue), "Max value must be greater than zero.");
        }

        _maxValue = maxValue;
        _currentValue = uint.MaxValue;
    }

    public int Next()
    {
        return (int)(Interlocked.Increment(ref _currentValue) % _maxValue);
    }
}