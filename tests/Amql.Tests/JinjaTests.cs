using System.Text.Json;
using Amql.Cli.Jinja;

namespace Amql.Tests;

/// <summary>The Jinja interpreter, held to what Python's Jinja renders in the environment transformers
/// compiles chat templates with (<c>fixtures/jinja_golden.json</c>, made by make_jinja_golden.py).</summary>
public sealed class JinjaTests
{
    private static Dictionary<string, object?> Vars(JsonElement e) => e.EnumerateObject().ToDictionary(p => p.Name, p => (object?)p.Value.Clone());

    [Fact]
    public void Templates_Render_As_Python_Renders_Them()
    {
        using var golden = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(AppContext.BaseDirectory, "fixtures", "jinja_golden.json")));
        var vars = Vars(golden.RootElement.GetProperty("vars"));
        var wrong = new List<string>();
        foreach (var c in golden.RootElement.GetProperty("cases").EnumerateArray())
        {
            string template = c.GetProperty("template").GetString()!;
            string? expected = c.GetProperty("text").GetString(), actual;
            try
            {
                actual = JinjaTemplate.Parse(template).Render(vars);
            }
            catch (JinjaException e)
            {
                actual = expected is null ? null : "! " + e.Message;
            }
            if (actual != expected)
            {
                wrong.Add($"{template}\n   expected {expected ?? "a failure"}\n   rendered {actual}");
            }
        }
        Assert.True(wrong.Count == 0, $"{wrong.Count} differ:\n" + string.Join("\n", wrong));
    }

    /// <summary>Real chat templates kept outside the repository (<c>AMQL_TEMPLATES</c>: a folder of the
    /// generator's files), each over its conversations.</summary>
    [Fact]
    public void Chat_Templates_Render_As_Transformers_Renders_Them()
    {
        if (Environment.GetEnvironmentVariable("AMQL_TEMPLATES") is not { Length: > 0 } dir)
        {
            return;
        }
        var wrong = new List<string>();
        int checkedCases = 0;
        foreach (string path in Directory.GetFiles(dir, "*.json"))
        {
            using var file = JsonDocument.Parse(File.ReadAllBytes(path));
            using var golden = JsonDocument.Parse(File.ReadAllBytes(path + ".golden"));
            JinjaTemplate template;
            try
            {
                template = JinjaTemplate.Parse(file.RootElement.GetProperty("template").GetString()!);
            }
            catch (JinjaException e)
            {
                wrong.Add($"{Path.GetFileName(path)}: not read: {e.Message}");
                continue;
            }
            foreach (var c in golden.RootElement.EnumerateArray())
            {
                checkedCases++;
                string? expected = c.GetProperty("text").GetString(), actual;
                try
                {
                    actual = template.Render(Vars(c.GetProperty("vars")));
                }
                catch (JinjaException e)
                {
                    actual = expected is null ? null : "! " + e.Message;
                }
                if (actual != expected)
                {
                    int at = Enumerable.Range(0, Math.Min(expected?.Length ?? 0, actual?.Length ?? 0)).FirstOrDefault(i => expected![i] != actual![i], -1);
                    string Near(string? s) => s is null ? "(fails)" : JsonSerializer.Serialize(s[Math.Max(0, Math.Min(at, s.Length) - 60)..Math.Min(s.Length, Math.Max(0, at) + 100)]);
                    wrong.Add($"{Path.GetFileName(path)}: expected {Near(expected)}\n   rendered {Near(actual)}");
                }
            }
        }
        Assert.True(wrong.Count == 0, $"{wrong.Count} of {checkedCases} differ:\n" + string.Join("\n", wrong.Take(40)));
        Assert.True(checkedCases > 0);
    }

    [Fact]
    public void What_A_Template_Can_Be_Given_Is_Known()
    {
        var reads = JinjaTemplate.Parse("{% set a = 1 %}{% macro m(x) %}{{ x }}{{ y }}{% endmacro %}{% for i, j in messages if z %}{{ i }}{{ loop.index }}{% endfor %}{{ a }}{{ m(tools) }}{{ enable_thinking is defined }}").Reads;
        Assert.Equal(["enable_thinking", "messages", "tools", "y", "z"], reads.Order());
    }

    [Theory]
    [InlineData("{% for i in range(99999) %}{% for j in range(99999) %}{{ i }}{% endfor %}{% endfor %}")]
    [InlineData("{% macro m() %}{{ m() }}{% endmacro %}{{ m() }}")]
    [InlineData("{{ range(100000000) }}")]
    public void A_Template_That_Never_Ends_Is_Stopped(string template) =>
        Assert.Throws<JinjaException>(() => JinjaTemplate.Parse(template).Render(new Dictionary<string, object?>()));
}
