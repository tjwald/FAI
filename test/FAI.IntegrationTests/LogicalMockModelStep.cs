namespace FAI.IntegrationTests;

public sealed class LogicalMockModelStep(float[][] outputs) :
    IStep<Tensor<long>[], TensorOutputs<float>>
{
    private int _callCount;

    public ValueTask<TensorOutputs<float>> ExecuteAsync(
        Tensor<long>[] input,
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

        return ValueTask.FromResult<TensorOutputs<float>>(new LogicalTensorOutputs(logits));
    }

    private sealed class LogicalTensorOutputs(Tensor<float> output) : TensorOutputs<float>
    {
        public override int Count => 1;

        public override ReadOnlyTensorSpan<float> GetOutput(int index)
            => index == 0 ? output.AsReadOnlyTensorSpan() : throw new ArgumentOutOfRangeException(nameof(index));

        public override void Dispose()
        {
        }
    }
}
