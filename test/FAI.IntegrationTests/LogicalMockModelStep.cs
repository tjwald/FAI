namespace FAI.IntegrationTests;

public sealed class LogicalMockModelStep(float[][] outputs) :
    IAllocatingStep<Tensor<long>[], Tensor<float>[]>,
    IBorrowedTensorProducer<Tensor<long>[], float>
{
    private int _callCount;

    public ValueTask<BatchLease<Tensor<float>[]>> RentOutputAsync(
        Tensor<long>[] input,
        CancellationToken cancellationToken = default)
    {
        int batchSize = checked((int)input[0].Lengths[0]);
        Tensor<float>[] output = [Tensor.CreateFromShape<float>([batchSize, outputs[0].Length])];
        return ValueTask.FromResult(new BatchLease<Tensor<float>[]>(output));
    }

    public ValueTask ExecuteAsync(
        Tensor<long>[] input,
        Tensor<float>[] output,
        CancellationToken cancellationToken = default)
    {
        for (int rowIndex = 0; rowIndex < output[0].Lengths[0]; rowIndex++)
        {
            float[] row = outputs[_callCount++ % outputs.Length];
            for (int columnIndex = 0; columnIndex < row.Length; columnIndex++)
            {
                output[0][rowIndex, columnIndex] = row[columnIndex];
            }
        }

        return ValueTask.CompletedTask;
    }

    public ValueTask ExecuteAsync<TOutput>(
        Tensor<long>[] input,
        TOutput output,
        IBorrowedTensorConsumer<float, TOutput> consumer,
        CancellationToken cancellationToken = default)
    {
        int batchSize = checked((int)input[0].Lengths[0]);
        Tensor<float> logits = Tensor.CreateFromShape<float>([batchSize, outputs[0].Length]);
        for (int rowIndex = 0; rowIndex < batchSize; rowIndex++)
        {
            float[] row = outputs[_callCount++ % outputs.Length];
            for (int columnIndex = 0; columnIndex < row.Length; columnIndex++)
            {
                logits[rowIndex, columnIndex] = row[columnIndex];
            }
        }

        consumer.Consume(logits.AsReadOnlyTensorSpan(), 0, output);
        return ValueTask.CompletedTask;
    }
}
