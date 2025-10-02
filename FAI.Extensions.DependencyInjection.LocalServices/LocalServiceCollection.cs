using System.Diagnostics.CodeAnalysis;

namespace FAI.Extensions.DependencyInjection.LocalServices;

public class LocalServiceCollection : ServiceCollection
{
    private readonly IServiceCollection _globalServices;
    private IServiceProvider? _serviceProvider;

    public LocalServiceCollection(IServiceCollection globalServices)
    {
        _globalServices = globalServices;
    }

    public LocalServiceCollection CopyToGlobal<TService, TImplementation>()
        where TService : class where TImplementation : class, TService
    {
        return AddGlobalHelper<TService, TImplementation>();
    }

    public LocalServiceCollection CopyToGlobal<TService>()
        where TService : class
    {
        return AddGlobalHelper<TService>();
    }

    private LocalServiceCollection AddGlobalHelper<TService>() where TService : class
    {
        _globalServices.AddSingleton(ServiceFactory<TService>);
        return this;
    }

    private LocalServiceCollection AddGlobalHelper<TService, TImplementation>() where TService : class where TImplementation : class, TService
    {
        _globalServices.AddSingleton<TService, TImplementation>(ServiceFactory<TImplementation>);
        return this;
    }

    private TService ServiceFactory<TService>(IServiceProvider _) where TService : notnull
    {
        InitServiceProvider();

        if (!typeof(TService).IsAbstract)
        {
            return ActivatorUtilities.CreateInstance<TService>(_serviceProvider);
        }

        ServiceDescriptor descriptor = this.First(x => x.ServiceType == typeof(TService));

        if (descriptor.ImplementationInstance is not null)
        {
            return (TService)descriptor.ImplementationInstance;
        }

        if (descriptor.ImplementationType is not null)
        {
            return (TService)ActivatorUtilities.CreateInstance(_serviceProvider, descriptor.ImplementationType)!;
        }

        if (descriptor.ImplementationFactory is not null)
        {
            return (TService)descriptor.ImplementationFactory(_serviceProvider);
        }

        throw new InvalidOperationException($"The descriptor for service: {typeof(TService).Name} is invalid");

    }

    [MemberNotNull(nameof(_serviceProvider))]
    private void InitServiceProvider()
    {
        if (_serviceProvider is not null) return;

        foreach (ServiceDescriptor serviceDescriptor in _globalServices)
        {
            ((IServiceCollection)this).Add(serviceDescriptor);
        }

        _serviceProvider = this.BuildServiceProvider();
    }
}
