using Microsoft.ML.OnnxRuntime;

namespace FAI.Onnx.Tests.Utils;

public static class OnnxTestModelFactory
{
    // A minimal ONNX model that casts INT64 input 'input' to FLOAT output 'output'. Shape [1, 3].
    // IR Version 6, Opset 11.
    private const string ConstantModelBase64 = "CAY6WwogCgVpbnB1dBIGb3V0cHV0IgRDYXN0KgkKAnRvGAGgAQISBHRlc3RaFwoFaW5wdXQSDgoMCAcSCAoCCAEKAggDYhgKBm91dHB1dBIOCgwIARIICgIIAQoCCANCBAoAEAs=";

    public static byte[] CreateMinimalModelBytes()
    {
        return Convert.FromBase64String(ConstantModelBase64);
    }

    public static string CreateTemporaryModelFile()
    {
        var path = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid()}.onnx");
        File.WriteAllBytes(path, CreateMinimalModelBytes());
        return path;
    }

    public static InferenceSession CreateSession()
    {
        return new InferenceSession(CreateMinimalModelBytes());
    }
}
