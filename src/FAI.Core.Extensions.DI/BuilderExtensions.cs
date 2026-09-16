using FAI.Core.Abstractions;
using Microsoft.Extensions.Options;

namespace FAI.Core.Extensions.DI;

public static class BuilderExtensions
{
    public static IServiceCollection AddConfigurationAndBind<TConfiguration>(this IServiceCollection services, string section) where TConfiguration : class
    {
        services.AddOptionsWithValidateOnStart<TConfiguration>()
            .BindConfiguration(section)
            .ValidateDataAnnotations();
        services.AddSingleton(sp => sp.GetRequiredService<IOptions<TConfiguration>>().Value);

        return services;
    }

    public static IServiceCollection AddPipelineInference<TInput, TOutput>(this IServiceCollection services)
    {
        services.AddSingleton<IInference<TInput, TOutput>, PipelineInference<TInput, TOutput>>();
        return services;
    }
}
