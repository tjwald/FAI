using Microsoft.Extensions.Options;

namespace FAI.Extensions.DependencyInjection.LocalServices;

public static class BuilderExtensions
{
    public static IServiceCollection AddConfigurationAndBind<TConfiguration>(this IServiceCollection services, string section) where TConfiguration : class
    {
        services.AddOptionsWithValidateOnStart<TConfiguration>()
            .BindConfiguration(section)
            .ValidateDataAnnotations();
        services.AddSingleton<TConfiguration>(sp => sp.GetRequiredService<IOptions<TConfiguration>>().Value);

        return services;
    }
}
