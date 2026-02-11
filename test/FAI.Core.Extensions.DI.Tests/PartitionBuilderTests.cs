using FAI.Core.Abstractions;
using FAI.Core.BatchSchedulers;
using FAI.Core.Configurations.PipelineBatchExecutors;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;

namespace FAI.Core.Extensions.DI.Tests;

public class PartitionBuilderTests
{
    private readonly IServiceCollection _services = new ServiceCollection();

    [Fact]
    public void WithSlicer_SetsSlicerFactory()
    {
        // Arrange
        var builder = new PartitionBatchExecutorBuilder<string, int>(_services);
        var mockSlicer = Substitute.For<IBatchSlicer<string>>();

        // Act
        builder.WithSlicer(_ => mockSlicer);
        var slicer = builder.BuildSlicer(_services.BuildServiceProvider());

        // Assert
        Assert.Same(mockSlicer, slicer);
    }

    [Fact]
    public void WithSchedular_SetsSchedularFactory()
    {
        // Arrange
        var builder = new PartitionBatchExecutorBuilder<string, int>(_services);
        var mockSchedular = Substitute.For<IBatchSchedular<string, int>>();

        // Act
        builder.WithSchedular(_ => mockSchedular);
        var schedular = builder.BuildSchedular(_services.BuildServiceProvider());

        // Assert
        Assert.Same(mockSchedular, schedular);
    }

    [Fact]
    public void WithSerialSchedular_RegistersOptionsAndSchedular()
    {
        // Arrange
        var config = new ConfigurationBuilder()
            .AddInMemoryCollection(new Dictionary<string, string?>
            {
                ["FAI:Serial:BatchSize"] = "10"
            })
            .Build();
        _services.AddSingleton<IConfiguration>(config);

        var builder = new PartitionBatchExecutorBuilder<string, int>(_services);

        // Act
        builder.WithSerialSchedular("FAI:Serial");
        var sp = _services.BuildServiceProvider();
        var schedular = builder.BuildSchedular(sp);

        // Assert
        Assert.IsType<SerialBatchSchedular<string, int>>(schedular);
        var options = sp.GetRequiredService<SerialBatchSchedularOptions>();
        Assert.Equal(10, options.BatchSize);
    }

    [Fact]
    public void WithParallelSchedular_RegistersOptionsAndSchedular()
    {
        // Arrange
        var config = new ConfigurationBuilder()
            .AddInMemoryCollection(new Dictionary<string, string?>
            {
                ["FAI:Parallel:MaxConcurrency"] = "4"
            })
            .Build();
        _services.AddSingleton<IConfiguration>(config);

        var builder = new PartitionBatchExecutorBuilder<string, int>(_services);

        // Act
        builder.WithParallelSchedular("FAI:Parallel");
        var sp = _services.BuildServiceProvider();
        var schedular = builder.BuildSchedular(sp);

        // Assert
        Assert.IsType<ParallelBatchSchedular<string, int>>(schedular);
        var options = sp.GetRequiredService<ParallelBatchSchedularOptions>();
        Assert.Equal(4, options.MaxConcurrency);
    }
}
