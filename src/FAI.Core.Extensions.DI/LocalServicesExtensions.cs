namespace FAI.Core.Extensions.DI;

public static class LocalServicesExtensions
{
    public static IServiceCollection AddLocalServices(this IServiceCollection services, Action<LocalServiceCollection> configure)
    {
        var scope = new LocalServiceCollection(services);
        configure(scope);
        return services;
    }
}
