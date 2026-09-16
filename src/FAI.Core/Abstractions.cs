// ReSharper disable once CheckNamespace
namespace FAI.Core.Abstractions
{
    /// <summary>
    /// Defines application-level inference operations.
    /// </summary>
    /// <typeparam name="TInput">The input type.</typeparam>
    /// <typeparam name="TOutput">The output type.</typeparam>
    public interface IInference<in TInput, TOutput>
    {
        /// <summary>
        /// Predicts output for the given input.
        /// </summary>
        Task<TOutput> Predict(TInput input);

        /// <summary>
        /// Predicts output into a caller-provided destination.
        /// </summary>
        Task Predict(TInput input, TOutput destination)
            => throw new NotSupportedException($"{GetType().Name} does not support destination-based prediction.");
    }

    /// <summary>
    /// Provides extension methods for inference operations.
    /// </summary>
    public static class InferenceExtensions
    {
        /// <summary>
        /// Predicts output for a single input by wrapping it in a batch.
        /// </summary>
        public static Task<TOutput> Predict<TInput, TOutput>(
            this IInference<ReadOnlyMemory<TInput>, TOutput> inference,
            TInput single)
            => inference.Predict((TInput[])[single]);
    }
}

namespace FAI.Core.Pipelines
{
    public interface IPipeline<in TInput, TOutput>
    {
        ValueTask<TOutput> ExecuteAsync(TInput input, CancellationToken cancellationToken = default);
    }

    public interface IDestinationPipeline<in TInput, TOutput> : IPipeline<TInput, TOutput>
    {
        ValueTask ExecuteAsync(
            TInput input,
            TOutput destination,
            CancellationToken cancellationToken = default);
    }

    public interface IReadOnlyIndexedBatch<TBatch>
    {
        int Count(TBatch batch);

        TBatch Slice(TBatch batch, Range range);

        BatchLease<TBatch> Gather(TBatch source, ReadOnlySpan<int> indices);
    }

    public interface IWritableIndexedBatch<TBatch> : IReadOnlyIndexedBatch<TBatch>
    {
        TBatch AllocateLike(TBatch template, int count);

        void Copy(TBatch source, TBatch destination);

        void Scatter(TBatch source, TBatch destination, ReadOnlySpan<int> destinationIndices);

        void PermuteInPlace(TBatch batch, Span<int> sourceToDestinationIndices);
    }

    public interface IBatchPartitioner<in TBatch>
    {
        IEnumerable<Range> Partition(TBatch batch);
    }

    public interface IIndexOrdering<in TBatch>
    {
        int[] CreateOrder(TBatch batch);
    }

    public interface IPartitionScheduler
    {
        ValueTask ExecuteAsync(
            IEnumerable<Range> ranges,
            Func<Range, CancellationToken, ValueTask> execute,
            CancellationToken cancellationToken = default);
    }

    public interface IBatchRoutingStrategy<TInput, TOutput>
    {
        List<BatchRoute<TInput, TOutput>> Route(TInput input);
    }
}
