namespace ML.Infra.Utilities;

public class CircularAtomicCounter
{
    private readonly int _maxValue;
    private int _currentValue;

    public CircularAtomicCounter(int maxValue)
    {
        if (maxValue <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(maxValue), "Max value must be greater than zero.");
        }

        _maxValue = maxValue;
        _currentValue = -1;
    }

    public int Next()
    {
        int initialValue, newValue;
        do
        {
            initialValue = _currentValue;
            newValue = (initialValue + 1) % _maxValue;
        } while (Interlocked.CompareExchange(ref _currentValue, newValue, initialValue) != initialValue);

        return newValue;
    }
}