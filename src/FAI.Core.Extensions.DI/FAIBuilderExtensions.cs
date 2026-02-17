using FAI.Core.Abstractions;
using FAI.Core.BatchSchedulers;
using FAI.Core.Configurations.PipelineBatchExecutors;
using FAI.Core.PipelineBatchExecutors;

namespace FAI.Core.Extensions.DI;

public static class FaiBuilderExtensions
{
    extension(IServiceCollection services)
    {
        public PipelineBuilder<TInput, TOutput> AddPipeline<TInput, TOutput>()
        {
            var pipelineBuilder = new PipelineBuilder<TInput, TOutput>(services);

            services.AddSingleton(sp => pipelineBuilder.Build(sp));

            return pipelineBuilder;
        }
    }

    extension<TInput, TOutput>(PipelineBuilder<TInput, TOutput> builder)
    {
        public PipelineBuilder<TInput, TOutput> UsePartitioning(Action<PartitionBatchExecutorBuilder<TInput, TOutput>> configure)
        {
            var partitionBuilder = new PartitionBatchExecutorBuilder<TInput, TOutput>(builder.ServiceCollection);
            configure(partitionBuilder);
            return builder.Use(
                (next, sp) => new PartitionPipelineBatchExecutor<TInput, TOutput>(partitionBuilder.BuildSchedular(sp), partitionBuilder.BuildSlicer(sp), next));
        }
    }

    extension<TInput, TOutput>(PartitionBatchExecutorBuilder<TInput, TOutput> builder)
    {
        public PartitionBatchExecutorBuilder<TInput, TOutput> WithSerialSchedular(string section)
        {
            builder.AddServices(serviceCollection =>
            {
                serviceCollection.AddConfigurationAndBind<SerialBatchSchedularOptions>(section);
                serviceCollection.AddSingleton<IBatchSchedular<TInput, TOutput>, SerialBatchSchedular<TInput, TOutput>>();
            });

            return builder.WithSchedular(sp => sp.GetRequiredService<IBatchSchedular<TInput, TOutput>>());
        }

        public PartitionBatchExecutorBuilder<TInput, TOutput> WithParallelSchedular(string section)
        {
            builder.AddServices(serviceCollection =>
            {
                serviceCollection.AddConfigurationAndBind<ParallelBatchSchedularOptions>(section);
                serviceCollection.AddSingleton<IBatchSchedular<TInput, TOutput>, ParallelBatchSchedular<TInput, TOutput>>();
            });

            return builder.WithSchedular(sp => sp.GetRequiredService<IBatchSchedular<TInput, TOutput>>());
        }
    }
}
