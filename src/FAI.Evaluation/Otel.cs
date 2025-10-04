using System.Diagnostics;
using System.Reflection;

namespace FAI.Evaluation;

internal static class Otel
{
    internal static ActivitySource Source { get; } = new(Assembly.GetAssembly(typeof(Otel))!.FullName!);
}
