using FAI.Onnx.Utils;

namespace FAI.Onnx.Tests.Utils;

public class CircularAtomicCounterTests
{
    [Fact]
    public void Next_ShouldReturnSequentialValues()
    {
        // Arrange
        var counter = new CircularAtomicCounter(3);

        // Act & Assert
        Assert.Equal(0, counter.Next());
        Assert.Equal(1, counter.Next());
        Assert.Equal(2, counter.Next());
        Assert.Equal(0, counter.Next());
    }

    [Fact]
    public void Constructor_ShouldThrow_WhenMaxValueIsZeroOrLess()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new CircularAtomicCounter(0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new CircularAtomicCounter(-1));
    }

    [Fact]
    public void Next_ShouldBeThreadSafe()
    {
        // Arrange
        const int maxValue = 10;
        const int iterations = 1000;
        const int threadCount = 10;
        var counter = new CircularAtomicCounter(maxValue);
        var results = new int[maxValue];
        var @lock = new Lock();

        // Act
        Parallel.For(0, threadCount, _ =>
        {
            for (int i = 0; i < iterations; i++)
            {
                int val = counter.Next();
                lock (@lock)
                {
                    results[val]++;
                }
            }
        });

        // Assert
        int totalIncrements = threadCount * iterations;
        Assert.Equal(totalIncrements, results.Sum());
        foreach (int count in results)
        {
            Assert.Equal(totalIncrements / maxValue, count);
        }
    }
}
