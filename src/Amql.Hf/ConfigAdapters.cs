using Amql.Vindex3;

namespace Amql.Hf;

/// <summary>
/// The config adapters: checkpoints whose config the generic reader does not
/// judge (<see cref="ModelConfig"/>) are read by an adapter of their own, which
/// describes the stack with the generic surfaces and a role map of its tensors.
/// The rest of the build asks here, never for a family by name.
/// </summary>
public static class ConfigAdapters
{
    /// <summary>Whether an adapter reads <paramref name="configPath"/>. Its config
    /// travels verbatim (export writes it back as it was).</summary>
    public static bool Reads(string configPath) => File.Exists(configPath) && Inkling2.Describes(configPath);

    /// <summary>The container spec the adapter that <see cref="Reads"/> the config writes.</summary>
    public static ContainerSpec Map(string modelId, string configPath, HfInventory inventory, ArchMapper.EncodeOptions options) =>
        ArchMapper.MapInkling2(modelId, configPath, inventory, options);
}

/// <summary>
/// Rewrites a system graph written by an older build into the surfaces this
/// build plans: each config adapter that recorded facts the planner no longer
/// reads (<see cref="ExecutionSurface.Unread"/>) rebuilds the generic surfaces
/// from them, exactly as a fresh encode writes them. Tensors are never touched.
/// </summary>
public static class GraphUpgrade
{
    /// <summary>The upgraded graph and the components rewritten (none when it is current).</summary>
    public static (SystemGraph Graph, IReadOnlyList<string> Upgraded) Apply(SystemGraph graph) => Inkling2.UpgradeGraph(graph);
}
