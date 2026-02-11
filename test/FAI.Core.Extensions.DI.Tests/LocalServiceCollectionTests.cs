using Microsoft.Extensions.DependencyInjection;

namespace FAI.Core.Extensions.DI.Tests;

public class LocalServiceCollectionTests
{
    [Fact]
    public void LocalServices_AreIsolatedFromGlobal()
    {
        // Arrange
        var globalServices = new ServiceCollection();

        // Act
        globalServices.AddLocalServices(local =>
        {
            local.AddSingleton<IMyService, MyService>();
        });

        var sp = globalServices.BuildServiceProvider();

        // Assert
        Assert.Null(sp.GetService<IMyService>());
    }

    [Fact]
    public void CopyToGlobal_MakesServiceAvailableInGlobal()
    {
        // Arrange
        var globalServices = new ServiceCollection();

        // Act
        globalServices.AddLocalServices(local =>
        {
            local.AddSingleton<IMyService, MyService>();
            local.CopyToGlobal<IMyService>();
        });

        var sp = globalServices.BuildServiceProvider();

        // Assert
        var service = sp.GetService<IMyService>();
        Assert.NotNull(service);
        Assert.IsType<MyService>(service);
    }

    [Fact]
    public void CopyToGlobal_PreservesLocalDependencies()
    {
        // Arrange
        var globalServices = new ServiceCollection();

        // Act
        globalServices.AddLocalServices(local =>
        {
            local.AddSingleton<IDependency, MyDependency>();
            local.AddSingleton<IMyService, ServiceWithDependency>();
            local.CopyToGlobal<IMyService>();
        });

        var sp = globalServices.BuildServiceProvider();

        // Assert
        var service = sp.GetService<IMyService>() as ServiceWithDependency;
        Assert.NotNull(service);
        Assert.NotNull(service.Dependency);

        // Dependency should NOT be in global
        Assert.Null(sp.GetService<IDependency>());
    }

    [Fact]
    public void LocalServices_CanResolveFromGlobal()
    {
        // Arrange
        var globalServices = new ServiceCollection();
        globalServices.AddSingleton<IGlobalService, GlobalService>();

        IMyService? localService = null;

        // Act
        globalServices.AddLocalServices(local =>
        {
            local.AddSingleton<IMyService, ServiceWithGlobalDependency>();
            local.CopyToGlobal<IMyService>();
        });

        var sp = globalServices.BuildServiceProvider();
        localService = sp.GetRequiredService<IMyService>();

        // Assert
        var service = Assert.IsType<ServiceWithGlobalDependency>(localService);
        Assert.NotNull(service.GlobalService);
    }

    private interface IMyService;
    private class MyService : IMyService;

    private interface IDependency;
    private class MyDependency : IDependency;

    private class ServiceWithDependency(IDependency dependency) : IMyService
    {
        public IDependency Dependency { get; } = dependency;
    }

    private interface IGlobalService;
    private class GlobalService : IGlobalService;

    private class ServiceWithGlobalDependency(IGlobalService globalService) : IMyService
    {
        public IGlobalService GlobalService { get; } = globalService;
    }
}
