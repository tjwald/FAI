using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Options;

namespace Example.SentimentInference.Model;

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
