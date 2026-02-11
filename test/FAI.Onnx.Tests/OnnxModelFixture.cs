using FAI.Onnx.Tests.Utils;

namespace FAI.Onnx.Tests;

public sealed class OnnxModelFixture : IDisposable
{
    public string ModelPath { get; }

    public OnnxModelFixture()
    {
        ModelPath = OnnxTestModelFactory.CreateTemporaryModelFile();
    }

    public void Dispose()
    {
        if (File.Exists(ModelPath))
        {
            File.Delete(ModelPath);
        }
    }
}
