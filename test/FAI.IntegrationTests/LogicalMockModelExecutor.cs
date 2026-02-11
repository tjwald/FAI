namespace FAI.IntegrationTests;

public class LogicalMockModelExecutor : IModelExecutor<long, float>
{
    private readonly float[][] _outputs;
    private int _callCount = 0;

    public LogicalMockModelExecutor(float[][] outputs)
    {
        _outputs = outputs;
    }

    public Task<Tensor<float>[]> RunAsync(Tensor<long>[] inputs)
    {
        var data = _outputs[_callCount % _outputs.Length];
        var output = Tensor.Create<float>(data, [(nint)data.Length]);
        _callCount++;
        return Task.FromResult(new[] { output });
    }

    public Task RunAsync(Tensor<long>[] inputs, Action<ReadOnlyTensorSpan<float>, int> postProcess)
    {
        int batchSize = (int)inputs[0].Lengths[0];
        int outputSize = _outputs[0].Length;

        float[] batchOutput = new float[batchSize * outputSize];
        for (int i = 0; i < batchSize; i++)
        {
            var row = _outputs[_callCount % _outputs.Length];
            _callCount++;
            row.AsSpan().CopyTo(batchOutput.AsSpan(i * outputSize));
        }

        var batchTensor = Tensor.Create<float>(batchOutput, [(nint)batchSize, (nint)outputSize]);
        postProcess(batchTensor, 0); // Assuming model has 1 output tensor

        return Task.CompletedTask;
    }
}
