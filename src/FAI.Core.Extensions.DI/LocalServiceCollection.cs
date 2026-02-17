using System.Diagnostics.CodeAnalysis;

namespace FAI.Core.Extensions.DI;

public class LocalServiceCollection : ServiceCollection
{
    private readonly IServiceCollection _globalServices;
    private IServiceProvider? _serviceProvider;

    public LocalServiceCollection(IServiceCollection globalServices)
    {
        _globalServices = globalServices;
    }

    public LocalServiceCollection CopyToGlobal<TService, TImplementation>(string? key = null)
        where TService : class where TImplementation : class, TService
    {
        return AddGlobalHelper<TService, TImplementation>(key);
    }

    public LocalServiceCollection CopyToGlobal<TService>(string? key = null)
        where TService : class
    {
        return AddGlobalHelper<TService>(key);
    }

    private LocalServiceCollection AddGlobalHelper<TService>(string? key = null) where TService : class
    {
        if (key is null)
        {
            _globalServices.AddSingleton(ServiceFactory<TService>);
        }
        else
        {
            _globalServices.AddKeyedSingleton(key, ServiceFactory<TService>);
        }

        return this;
    }

    private LocalServiceCollection AddGlobalHelper<TService, TImplementation>(string? key = null) where TService : class where TImplementation : class, TService
    {
        if (key is null)
        {
            _globalServices.AddSingleton<TService, TImplementation>(ServiceFactory<TImplementation>);
        }
        else
        {
            _globalServices.AddKeyedSingleton<TService, TImplementation>(key, ServiceFactory<TImplementation>);
        }

        return this;
    }

    private TService ServiceFactory<TService>(IServiceProvider sp) where TService : notnull
    {
        return ServiceFactory<TService>(sp, null);
    }

    private TService ServiceFactory<TService>(IServiceProvider sp, object? o) where TService : notnull
    {
        InitServiceProvider();

        if (!typeof(TService).IsAbstract)
        {
            return ActivatorUtilities.CreateInstance<TService>(_serviceProvider);
        }

        ServiceDescriptor? descriptor = null;
        if (o is not null)
        {
            descriptor = this.FirstOrDefault(x => x.ImplementationType == typeof(TService) && (!x.IsKeyedService || x.ServiceKey == o));
        }

        descriptor ??= this.First(x => x.ServiceType == typeof(TService));

        if (descriptor.IsKeyedService)
        {
            if (descriptor.KeyedImplementationInstance is not null)
            {
                return (TService)descriptor.KeyedImplementationInstance;
            }

            if (descriptor.KeyedImplementationFactory is not null)
            {
                return (TService)descriptor.KeyedImplementationFactory(_serviceProvider, o);
            }
        }

        if (descriptor.ImplementationInstance is not null)
        {
            return (TService)descriptor.ImplementationInstance;
        }

        if (descriptor.ImplementationType is not null)
        {
            return (TService)ActivatorUtilities.CreateInstance(_serviceProvider, descriptor.ImplementationType);
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
