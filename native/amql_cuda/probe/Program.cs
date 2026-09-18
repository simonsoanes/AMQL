using Amql.Inference;
using Amql.Vindex3;
using System.Diagnostics;

Console.WriteLine("sessionprobe — time the real prefill+steps on the merged container");
Environment.SetEnvironmentVariable("AMQL_GPU", "1");
Environment.SetEnvironmentVariable("AMQL_WEIGHTS", "mxfp4");
Environment.SetEnvironmentVariable("AMQL_F32_CACHE_GB", "60");
CudaShim.Reset();

using var container = Vindex3Container.Open(@"D:\Dev\AMQL\containers\Qwen3.8-27B-merged");
using var store = container.CreateOperandStore();
var plan = Planner.Plan(container, "target", store);
var sw = Stopwatch.StartNew();
var session = new DecodeSession(plan, store, workingSet: WeightWorkingSet.Mxfp4);
Console.WriteLine($"session ready at {sw.Elapsed.TotalSeconds:F1}s failed={CudaShim.DeviceFailed}");

sw.Restart();
var first = session.Prefill(new[] { 1, 14556, 1 }).FirstRow().ToArray();
Console.WriteLine($"prefill (m=3) at {sw.Elapsed.TotalSeconds:F1}s failed={CudaShim.DeviceFailed} weights={CudaShim.DeviceWeightCount}");

for (int s = 0; s < 10; s++)
{
    sw.Restart();
    var row = session.Step(3966).FirstRow().ToArray();
    Console.WriteLine($"step {s} at {sw.Elapsed.TotalSeconds:F2}s failed={CudaShim.DeviceFailed}");
}
Console.WriteLine("sessionprobe done");