using FAI.Core.Abstractions;

namespace FAI.Core.BatchSlicers;

public sealed record FixedSizeBatchSlicerOptions(int BatchSize)
{
    public FixedSizeBatchSlicerOptions() : this(0) { }
}

public class FixedSizeBatchSlicer<TInput> : IBatchSlicer<TInput>
{
    private readonly FixedSizeBatchSlicerOptions _options;

    public FixedSizeBatchSlicer(FixedSizeBatchSlicerOptions options)
    {
        _options = options;
    }

    public IEnumerable<Range> Slice(ReadOnlyMemory<TInput> inputs)
    {
        int i = 0;
        for (; i < inputs.Length - _options.BatchSize; i++)
        {
            yield return new Range(i, i + _options.BatchSize);
        }

        if (i < inputs.Length)
        {
            yield return new Range(i, inputs.Length);
        }
    }
}
