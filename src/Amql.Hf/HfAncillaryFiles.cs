namespace Amql.Hf;

/// <summary>
/// The ancillary files an HF checkpoint ships that a container has to keep in
/// order to stay re-exportable: the chat template, the processor configs a
/// multi-modal runtime needs, and the generation/special-token config.
/// <para>
/// Deliberately excluded: the weights (handled by the encoder),
/// <c>config.json</c> (regenerated from the container's own arch facts, so a
/// stale copy would be actively wrong), <c>tokenizer.json</c> (a merge
/// rewrites it rather than copying it), and <c>vocab.json</c> /
/// <c>merges.txt</c> / <c>tokenizer.model</c>, which duplicate the vocabulary
/// already carried in <c>tokenizer.json</c> — this build only supports the HF
/// tokenizers format, so a second source of truth would just drift.
/// </para>
/// </summary>
public static class HfAncillaryFiles
{
    /// <summary>Copied in this order; missing files are skipped silently, since
    /// a text-only checkpoint legitimately has no preprocessor config.</summary>
    public static readonly IReadOnlyList<string> Names = new[]
    {
        "chat_template.jinja",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "generation_config.json",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
        "processor_config.json",
        // Von's option-marker calibration: the fitted temperature map, and the
        // independent_options flag its weights must be run with.
        "marker_calibration.json",
        "calibration.json",
    };

    /// <summary>Copies whichever of <see cref="Names"/> exist from
    /// <paramref name="sourceDir"/> into <paramref name="destDir"/>, creating
    /// the destination if needed, and returns the names actually copied.
    /// Overwrites by default: re-exporting into a directory that already holds
    /// a previous export is normal, and these are small config files refreshed
    /// from the container rather than anything a user edits in place.</summary>
    public static IReadOnlyList<string> CopyInto(string sourceDir, string destDir, bool overwrite = true)
    {
        var copied = new List<string>();
        foreach (string name in Names.Concat(RemoteCodeModules(sourceDir)))
        {
            string source = Path.Combine(sourceDir, name);
            if (!File.Exists(source))
            {
                continue;
            }
            Directory.CreateDirectory(destDir);
            File.Copy(source, Path.Combine(destDir, name), overwrite);
            copied.Add(name);
        }
        return copied;
    }

    /// <summary>Which of <see cref="Names"/> are present in
    /// <paramref name="dir"/>, for reporting what an export will carry.</summary>
    public static IReadOnlyList<string> Present(string dir)
        => Names.Concat(RemoteCodeModules(dir)).Where(name => File.Exists(Path.Combine(dir, name))).ToArray();

    /// <summary>The remote-code modules a checkpoint's <c>auto_map</c> names
    /// load from (<c>modeling_*.py</c> / <c>configuration_*.py</c>, the HF
    /// convention). They are carried byte for byte and never executed here:
    /// without them an exported checkpoint whose config points at a custom
    /// class (a masked diffusion A2D model, say) cannot be loaded.</summary>
    private static IEnumerable<string> RemoteCodeModules(string dir) =>
        Directory.Exists(dir)
            ? Directory.EnumerateFiles(dir, "*.py")
                .Select(Path.GetFileName)
                .Where(n => n!.StartsWith("modeling_", StringComparison.Ordinal) ||
                            n.StartsWith("configuration_", StringComparison.Ordinal))
                .Order(StringComparer.Ordinal)!
            : Enumerable.Empty<string>();
}
