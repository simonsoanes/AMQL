using Amql.Entail;
using Amql.Hf;
using Amql.Inference;
using Amql.Vindex3;
using Xunit;

namespace Amql.Tests;

/// <summary>A program as a blank model beside a partner: what its verdicts mean,
/// and that the two stay on the same text.</summary>
public class EntailAdapterTests
{
    private static EntailAdapterManifest Manifest(params EntailAdapterOutput[] outputs) =>
        new("test", "00", new Dictionary<int, int> { [5] = 2 }, outputs);

    private static EntailAdapterOutput Output(int row, EntailOutputKind kind, params int[] tokens) => new(row, kind, tokens);

    private static float[] Lanes(int count, params int[] set)
    {
        var logits = new float[count];
        foreach (int i in set)
        {
            logits[i] = 1;
        }
        return logits;
    }

    [Fact]
    public void All_Zero_Logits_Are_Silence()
    {
        var verdict = EntailVerdict.Read(Manifest(Output(0, EntailOutputKind.Say, 7)), new float[1]);
        Assert.True(verdict.Silent);
        Assert.Equal((2, EntailTokenSource.Partner), verdict.Choose(new[] { 0f, 1f, 3f }));
    }

    [Fact]
    public void A_Token_That_Must_Come_Next_Overrides_The_Partner()
    {
        var verdict = EntailVerdict.Read(Manifest(Output(0, EntailOutputKind.Say, 1)), Lanes(1, 0));
        Assert.Equal((1, EntailTokenSource.Entail), verdict.Choose(new[] { 0f, -50f, 3f }));
    }

    [Fact]
    public void The_First_Emit_Counts_And_Every_Mask_Applies()
    {
        var manifest = Manifest(
            Output(0, EntailOutputKind.End, 1),
            Output(1, EntailOutputKind.Say, 2),
            Output(2, EntailOutputKind.Forbid, 3));
        var verdict = EntailVerdict.Read(manifest, Lanes(3, 0, 1, 2));
        Assert.Null(verdict.Say);
        Assert.False(verdict.Permits(1));
        Assert.False(verdict.Permits(3));
        Assert.True(verdict.Permits(2));
    }

    [Fact]
    public void A_Mask_Steers_Only_When_It_Changes_The_Choice()
    {
        var verdict = EntailVerdict.Read(Manifest(Output(0, EntailOutputKind.Only, 0, 1)), Lanes(1, 0));
        Assert.Equal((1, EntailTokenSource.Partner), verdict.Choose(new[] { 0f, 2f, 1f }));
        Assert.Equal((1, EntailTokenSource.Narrowed), verdict.Choose(new[] { 0f, 2f, 9f }));
    }

    [Fact]
    public void Adapters_Combine_In_Order()
    {
        var one = EntailVerdict.Read(Manifest(Output(0, EntailOutputKind.Say, 1), Output(1, EntailOutputKind.Only, 1, 2)), Lanes(2, 0, 1));
        var two = EntailVerdict.Read(Manifest(Output(0, EntailOutputKind.Say, 2), Output(1, EntailOutputKind.Only, 2, 3)), Lanes(2, 0, 1));
        var merged = EntailVerdict.Merge(new[] { EntailVerdict.Silence, one, two });
        Assert.Equal(1, merged.Say);
        Assert.True(merged.Conflict);
        Assert.Equal(new[] { 2 }, merged.Allowed!.ToArray());
        Assert.True(EntailVerdict.Merge(new[] { EntailVerdict.Silence, EntailVerdict.Silence }).Silent);
    }

    [Fact]
    public void Rules_That_Leave_Nothing_Set_The_Partner_Free()
    {
        var verdict = EntailVerdict.Read(
            Manifest(Output(0, EntailOutputKind.Only, 1), Output(1, EntailOutputKind.Forbid, 1)), Lanes(2, 0, 1));
        Assert.Equal((2, EntailTokenSource.Partner), verdict.Choose(new[] { 0f, 1f, 3f }));
        Assert.True(verdict.Conflict);
    }

    /// <summary>A partner with nothing to say, and its tokenizer.</summary>
    private static (string Checkpoint, string Container) Partner(string root)
    {
        var checkpoint = Path.Combine(root, "partner.checkpoint");
        var container = Path.Combine(root, "partner");
        EntailWeights.WriteFresh(EntailCompiler.Compile("state a, b\ntoken \"x\" means a = 1\nentail b when a"), checkpoint);
        ModelToContainer.Encode(checkpoint, container, "partner");
        return (checkpoint, container);
    }

    private static string Adapter(string root, string name, EntailProgram program, string tokenizer)
    {
        var checkpoint = Path.Combine(root, name + ".checkpoint");
        var container = Path.Combine(root, name);
        var (_, manifest) = EntailWeights.WriteAdapter(program, tokenizer, checkpoint);
        ModelToContainer.Encode(checkpoint, container, name);
        manifest.Save(container);
        return container;
    }

    [Fact]
    public void A_Manifest_Survives_The_Disk_And_Refuses_Another_Tokenizer()
    {
        using var dir = new TempDir();
        var manifest = Manifest(Output(0, EntailOutputKind.Forbid, 3, 4));
        manifest.Save(dir.Path);
        var loaded = EntailAdapterManifest.Load(dir.Path);
        Assert.Equal(2, loaded.Row(5));
        Assert.Equal(0, loaded.Row(6));
        Assert.Equal(EntailOutputKind.Forbid, loaded.Outputs[0].Kind);
        Assert.Equal(new[] { 3, 4 }, loaded.Outputs[0].Tokens);

        File.WriteAllText(Path.Combine(dir.Path, "tokenizer.json"), "{}");
        Assert.Throws<EntailException>(() => loaded.RequirePartner(dir.Path));
    }

    /// <summary>The whole path: both models read the same tokens, the program
    /// answers, every token of the answer is marked as its own, and it lets go.</summary>
    [Fact]
    public void An_Adapter_Answers_Beside_A_Partner_And_Lets_Go()
    {
        using var dir = new TempDir();
        var (checkpoint, partnerDir) = Partner(dir.Path);
        var adapterDir = Adapter(dir.Path, "maths",
            EntailCompiler.Compile(File.ReadAllText(EntailTests.RepoFile("scripts", "entail", "mathematics.entail.txt"))), checkpoint);

        var tokenizer = HfTokenizer.FromModelDir(checkpoint);
        using var container = Vindex3Container.Open(partnerDir);
        using var store = container.CreateOperandStore();
        var plan = Planner.Plan(container, "target", store);
        using var adapter = EntailAdapterRunner.Open(adapterDir, checkpoint);
        var side = new SideBySideSession(new DecodeSession(plan, store, null, WeightWorkingSet.ResidentF32), new IEntailAdapter[] { adapter });

        side.Prefill(tokenizer.EncodeToIds(" 12+34=").ToArray());
        var reply = new List<int>();
        for (int i = 0; i < 8; i++)
        {
            var (token, source) = side.Choose();
            if (token <= 1)
            {
                break;
            }
            Assert.Equal(EntailTokenSource.Entail, source);
            reply.Add(token);
            side.Step(token);
        }
        Assert.Equal("46", tokenizer.DecodeText(reply));

        side.Reset();
        side.Prefill(tokenizer.EncodeToIds(" no sums here").ToArray());
        Assert.True(side.Verdict.Silent);

        var alone = new SideBySideSession(new DecodeSession(plan, store, null, WeightWorkingSet.ResidentF32), Array.Empty<IEntailAdapter>());
        alone.Prefill(tokenizer.EncodeToIds(" 12+34=").ToArray());
        Assert.True(alone.Verdict.Silent);
        Assert.Equal(0, alone.Choose().Token);
    }

    private static string Maths => File.ReadAllText(EntailTests.RepoFile("scripts", "entail", "mathematics.entail.txt"));

    /// <summary>The ring has to give the same answers as holding every token,
    /// so the text runs well past the program's window, and past it again.</summary>
    [Fact]
    public void Running_The_Statements_Agrees_With_The_Interpreter_At_Every_Token()
    {
        using var dir = new TempDir();
        var (checkpoint, _) = Partner(dir.Path);
        EntailSparseAdapter.Write(Maths, checkpoint, Path.Combine(dir.Path, "sparse"));
        using var sparse = EntailSparseAdapter.Open(Path.Combine(dir.Path, "sparse"), checkpoint);
        var tokenizer = HfTokenizer.FromModelDir(checkpoint);
        var program = EntailCompiler.Compile(Maths);

        string text = string.Concat(Enumerable.Range(0, 40).Select(i => $" {i * 37}+{i * 91 + 5}= and {i + 2}*{i + 3}={(i + 2) * (i + 3)} then"));
        var ids = tokenizer.EncodeToIds(text).ToArray();
        var lanes = EntailInterpreter.Run(program, ids.Select(id => tokenizer.DecodeText(new List<int> { id })).ToList());
        int spoke = 0;
        for (int t = 0; t < ids.Length; t++)
        {
            var verdict = sparse.Step(ids[t]);
            var expected = EntailInterpreter.Output(program, lanes[t]);
            string? said = verdict.Say is { } id ? tokenizer.DecodeText(new List<int> { id }) : verdict.Silent ? null : string.Empty;
            Assert.Equal(expected, said);
            spoke += said is null ? 0 : 1;
        }
        Assert.True(spoke > 40);
    }

    [Fact]
    public void Statements_And_Dense_Weights_Give_The_Same_Verdicts()
    {
        using var dir = new TempDir();
        var (checkpoint, _) = Partner(dir.Path);
        var denseDir = Adapter(dir.Path, "dense", EntailCompiler.Compile(Maths), checkpoint);
        EntailSparseAdapter.Write(Maths, checkpoint, Path.Combine(dir.Path, "sparse"));
        using var dense = EntailAdapterRunner.Load(denseDir, checkpoint);
        using var sparse = EntailAdapterRunner.Load(Path.Combine(dir.Path, "sparse"), checkpoint);
        Assert.IsType<EntailAdapterRunner>(dense);
        Assert.IsType<EntailSparseAdapter>(sparse);

        var ids = HfTokenizer.FromModelDir(checkpoint).EncodeToIds(" 12+34=46 and 7*8=56, not 99-1 but 144/12=12").ToArray();
        dense.Prefill(ids[..1]);
        sparse.Prefill(ids[..1]);
        foreach (int id in ids[1..])
        {
            var a = dense.Step(id);
            var b = sparse.Step(id);
            Assert.Equal(a.Say, b.Say);
            Assert.Equal(a.Silent, b.Silent);
            Assert.Equal(a.Banned.Order(), b.Banned.Order());
        }
    }

    [Fact]
    public void Readers_Of_One_Program_Keep_Their_Own_Place()
    {
        using var dir = new TempDir();
        var (checkpoint, _) = Partner(dir.Path);
        EntailSparseAdapter.Write(Maths, checkpoint, Path.Combine(dir.Path, "sparse"));
        using var one = EntailSparseAdapter.Open(Path.Combine(dir.Path, "sparse"), checkpoint);
        var two = one.Sibling();
        var tokenizer = HfTokenizer.FromModelDir(checkpoint);
        one.Prefill(tokenizer.EncodeToIds(" 12+3").ToArray());
        var other = two.Prefill(tokenizer.EncodeToIds(" 2+2=").ToArray());
        var first = one.Step(tokenizer.EncodeToIds("4")[0]);
        Assert.Equal(tokenizer.EncodeToIds("4")[0], other.Say);
        Assert.True(first.Silent);
        one.Reset();
        Assert.Equal(tokenizer.EncodeToIds("4")[0], one.Prefill(tokenizer.EncodeToIds(" 2+2=").ToArray()).Say);
    }

    [Fact]
    public void A_Mask_Reaches_The_Sampler_Through_The_Weights()
    {
        using var dir = new TempDir();
        var (checkpoint, _) = Partner(dir.Path);
        var program = EntailCompiler.Compile("state a, b\ntoken \"x\" means a = 1\nentail b when a\nonly \"y\", \"z\" when b");
        var adapterDir = Adapter(dir.Path, "mask", program, checkpoint);
        var tokenizer = HfTokenizer.FromModelDir(checkpoint);
        using var adapter = EntailAdapterRunner.Open(adapterDir, checkpoint);

        var verdict = adapter.Prefill(tokenizer.EncodeToIds("x").ToArray());
        Assert.False(verdict.Silent);
        Assert.True(verdict.Permits(tokenizer.EncodeToIds("y")[0]));
        Assert.False(verdict.Permits(tokenizer.EncodeToIds("q")[0]));

        adapter.Reset();
        Assert.True(adapter.Prefill(tokenizer.EncodeToIds("q").ToArray()).Silent);
    }
}
