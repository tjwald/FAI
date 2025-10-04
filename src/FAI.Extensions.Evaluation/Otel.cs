using System.Diagnostics;

namespace FAI.Extensions.Evaluation;

internal static class Otel
{
    internal static ActivitySource Source { get; } = new("fai.evaluation");
}
