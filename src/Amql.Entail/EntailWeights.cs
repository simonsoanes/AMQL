using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using Amql.Safetensors;

namespace Amql.Entail;

/// <summary>What a compile reports about the model it wrote.</summary>
public sealed record EntailBuildReport(
    string Directory, int Hidden, int Layers, int BlockLayers, int LanesUsed, int LanesPeak,
    int Neurons, int Heads, double MinAttentionGap, int TokensBound, IReadOnlyList<string> TokensMissing);

/// <summary>
/// Turns an <see cref="EntailProgram"/> into tensors, in the Qwen3.5 layer
/// shape (RMSNorm with a 1+w weight, gated attention with per-head QK norm,
/// partial RoPE, SwiGLU). The compiled block reads and writes only its own
/// lanes, so it is the same set of layers whether it stands alone in an empty
/// model or is grafted in front of a host's.
/// <para>
/// Three facts make a stack of RMSNorms behave like exact integer logic. A
/// constant "ballast", far larger than everything else, fixes the norm's
/// divisor. It is spread over just enough dimensions that the mean of squares
/// is a power of four, so the divisor is a power of two; everything else is
/// too small to survive the rounding back to float, so a lane reads back as
/// precisely the number written to it, and integers stay integers from layer
/// to layer with no drift at all. And a lane's norm weight, also a power of
/// two, undoes the divisor, while a host dimension's is zero, so nothing else
/// can reach a lane.
/// </para>
/// </summary>
public static class EntailWeights
{
    /// <summary>Raw value of each ballast dimension; a power of two, so
    /// bf16-exact.</summary>
    public const double Ballast = 1048576;
    private const int Rotary = 64;        // rotary dims of a head (256 × 0.25)
    private const int Pairs = 16;         // frequency pairs an offset uses
    private const double RopeTheta = 1e7;
    private const double NormGain = 28;   // q/k norm weight: sharpens attention
    private const double OutputGate = 30; // sigmoid(30) = 1
    private const int QDim = EntailProgram.HeadsPerLayer * EntailProgram.HeadDim;
    private const int KvDim = EntailProgram.KvGroups * EntailProgram.HeadDim;
    private const int HeadsPerGroup = EntailProgram.HeadsPerLayer / EntailProgram.KvGroups;

    /// <summary>Lane → dimension assignment by liveness: a lane nothing reads
    /// any more is cleared by one neuron and its dimension is reused.</summary>
    private sealed class Allocation
    {
        public int[] Dim = Array.Empty<int>();             // lane → dim index, -1 if unused
        public int Peak;
        public int Used;
        public List<(int Layer, int Lane)> Clears = new(); // neuron that zeroes a dead lane
        public List<int> LiveAtEnd = new();
    }

    private static Allocation Allocate(EntailProgram p, int firstDim, bool keepEmitsLive)
    {
        int n = p.Lanes.Count;
        var firstWrite = new int[n];
        var lastRead = new int[n];
        var written = new bool[n];
        Array.Fill(firstWrite, int.MaxValue);
        Array.Fill(lastRead, -1);
        foreach (var writes in p.Tokens.Values)
        {
            foreach (var lane in writes.Keys)
            {
                firstWrite[lane] = -1;
                written[lane] = true;
            }
        }
        void Read(Lin l, int time)
        {
            foreach (var lane in l.Terms.Keys)
            {
                lastRead[lane] = Math.Max(lastRead[lane], time);
            }
        }
        foreach (var op in p.Ops)
        {
            if (op is Neuron neuron)
            {
                int t = 2 * neuron.Layer + 1;
                Read(neuron.Gate, t);
                Read(neuron.Up, t);
                firstWrite[neuron.Dest] = Math.Min(firstWrite[neuron.Dest], t);
                written[neuron.Dest] = true;
            }
            else if (op is Head head)
            {
                int t = 2 * head.Layer;
                foreach (var (dest, source) in head.Copies)
                {
                    Read(source, t);
                    firstWrite[dest] = Math.Min(firstWrite[dest], t);
                    written[dest] = true;
                }
                if (head.Pointer is { } pointer)
                {
                    foreach (var (lane, _) in pointer)
                    {
                        lastRead[lane] = Math.Max(lastRead[lane], t);
                    }
                }
            }
        }

        int end = 2 * p.Layers + 1;
        foreach (int lane in p.Emits.Select(e => e.Lane).Concat(p.Masks.Select(m => m.Lane)))
        {
            lastRead[lane] = keepEmitsLive ? int.MaxValue : end;
            if (keepEmitsLive)
            {
                // With a tied head an output lane is an embedding column, so
                // it may never share a dimension with an input lane.
                firstWrite[lane] = -1;
            }
        }

        var result = new Allocation { Dim = new int[n] };
        Array.Fill(result.Dim, -1);
        var free = new PriorityQueue<int, int>();          // dim, usable from time
        var order = Enumerable.Range(0, n).Where(l => written[l] && lastRead[l] >= 0)
            .OrderBy(l => firstWrite[l]).ThenBy(l => l).ToList();
        int nextDim = firstDim, live = 0;
        var active = new PriorityQueue<int, int>();        // lane by last read
        foreach (int lane in order)
        {
            while (active.TryPeek(out int old, out int until) && until < firstWrite[lane] - 1)
            {
                active.Dequeue();
                live--;
                // Cleared by the FFN that runs at or just after the last read.
                int clearLayer = lastRead[old] / 2;
                int clearTime = 2 * clearLayer + 1;
                if (clearTime < lastRead[old])
                {
                    clearLayer++;
                    clearTime += 2;
                }
                free.Enqueue(result.Dim[old], clearTime);
                result.Clears.Add((clearLayer, old));
            }
            if (free.TryPeek(out int dim, out int from) && from < firstWrite[lane])
            {
                free.Dequeue();
                result.Dim[lane] = dim;
            }
            else
            {
                result.Dim[lane] = nextDim++;
            }
            active.Enqueue(lane, lastRead[lane]);
            live++;
            result.Peak = Math.Max(result.Peak, live);
        }
        while (active.TryDequeue(out int lane, out _))
        {
            if (!p.Lanes[lane].Emits)
            {
                result.LiveAtEnd.Add(lane);
            }
        }
        result.Used = nextDim - firstDim;
        return result;
    }

    // ── attention geometry ──────────────────────────────────────────────────

    private static double Omega(int pair) => Math.Pow(RopeTheta, -2.0 * pair / Rotary);

    /// <summary>A query, before rotation, whose score against the constant
    /// key peaks at exactly <paramref name="offset"/> tokens back.</summary>
    private static double[] OffsetQuery(int offset)
    {
        var q = new double[EntailProgram.HeadDim];
        for (int i = 0; i < Pairs; i++)
        {
            double w = Omega(i);
            q[i] = Math.Cos(w * offset);
            q[i + Rotary / 2] = -Math.Sin(w * offset);
        }
        return q;
    }

    private static double Score(double[] q, int delta)
    {
        // q·R(δ)k for the constant key k, in the units attention will see.
        double qn = 0;
        foreach (var v in q)
        {
            qn += v * v;
        }
        double kScale = Math.Sqrt(EntailProgram.HeadDim / (double)Pairs) * NormGain;
        double qScale = Math.Sqrt(EntailProgram.HeadDim / qn) * NormGain;
        double score = 0;
        for (int i = 0; i < Pairs; i++)
        {
            double w = Omega(i) * delta;
            score += q[i] * Math.Cos(w) - q[i + Rotary / 2] * Math.Sin(w);
        }
        return score * qScale * kScale / Math.Sqrt(EntailProgram.HeadDim);
    }

    /// <summary>The smallest margin by which any head's intended token beats
    /// every other token, over the whole context. Compile-time proof that
    /// attention is a lookup, not a blend.</summary>
    public static double AttentionGap(EntailProgram p, int context)
    {
        double gap = double.MaxValue;
        foreach (var head in p.Ops.OfType<Head>())
        {
            var targets = head.Pointer?.Select(x => x.Offset).ToArray() ?? new[] { head.Offset };
            foreach (int target in targets)
            {
                var q = OffsetQuery(target);
                double peak = Score(q, target);
                for (int delta = 0; delta < context; delta++)
                {
                    if (delta != target)
                    {
                        gap = Math.Min(gap, peak - Score(q, delta));
                    }
                }
            }
        }
        return gap;
    }

    // ── the block ───────────────────────────────────────────────────────────

    private sealed class Layer
    {
        public required float[] Q, K, V, O, Gate, Up, Down, InNorm, PostNorm, QNorm, KNorm;
    }

    private sealed class Geometry
    {
        public required int Hidden, Intermediate, Ballast, BallastDims, Shift;
        public required Allocation Lanes;
        public required double[] Scale;   // lane → raw per unit, this target
        public required bool[] HostDim;   // dims the block must not read
    }

    /// <summary>Splits hidden = n · 4^shift with shift as large as possible:
    /// n ballast dimensions make the mean of squares Ballast² / 4^shift, so
    /// the norm's divisor is the power of two Ballast / 2^shift.</summary>
    private static (int Dims, int Shift) BallastShape(int hidden)
    {
        int n = hidden, shift = 0;
        while (n % 4 == 0)
        {
            n /= 4;
            shift++;
        }
        return (n, shift);
    }

    /// <summary>The smallest width holding <paramref name="fixedDims"/> plus
    /// its own ballast.</summary>
    private static int WidthFor(int fixedDims)
    {
        for (int hidden = RoundUp(fixedDims + 1, 64); ; hidden += 64)
        {
            if (fixedDims + BallastShape(hidden).Dims <= hidden)
            {
                return hidden;
            }
        }
    }

    private static void ReadInto(float[] matrix, int row, int hidden, Lin form, Geometry g, double factor = 1)
    {
        int at = row * hidden;
        matrix[at + g.Ballast] += (float)(form.Const * factor / Math.Pow(2, g.Shift));
        foreach (var (lane, c) in form.Terms)
        {
            int dim = g.Lanes.Dim[lane];
            if (dim >= 0)
            {
                matrix[at + dim] += (float)(c * factor / g.Scale[lane]);
            }
        }
    }

    private static float[] NormRow(Geometry g, double laneGamma)
    {
        var w = new float[g.Hidden];
        for (int d = 0; d < g.Hidden; d++)
        {
            bool spare = d > g.Ballast && d < g.Ballast + g.BallastDims;
            double gamma = g.HostDim[d] || spare ? 0 : d == g.Ballast ? 1 : laneGamma;
            w[d] = (float)(gamma - 1);   // the family applies (1 + w)
        }
        return w;
    }

    private static Layer BuildLayer(EntailProgram p, Geometry g, int index,
        IEnumerable<(Lin Gate, Lin Up, int Dest, double Coef)> extra,
        IEnumerable<(double GateConst, int Lane, bool ClearBallast)> clears)
    {
        int h = g.Hidden, inter = g.Intermediate;
        var layer = new Layer
        {
            Q = new float[2 * QDim * h],
            K = new float[KvDim * h],
            V = new float[KvDim * h],
            O = new float[h * QDim],
            Gate = new float[inter * h],
            Up = new float[inter * h],
            Down = new float[h * inter],
            InNorm = NormRow(g, Ballast / Math.Pow(2, g.Shift)),
            PostNorm = NormRow(g, Ballast / Math.Pow(2, g.Shift)),
            QNorm = Enumerable.Repeat((float)(NormGain - 1), EntailProgram.HeadDim).ToArray(),
            KNorm = Enumerable.Repeat((float)(NormGain - 1), EntailProgram.HeadDim).ToArray(),
        };

        // Heads: each group shares one constant key and one value table.
        var heads = p.Ops.OfType<Head>().Where(x => x.Layer == index).ToList();
        var slots = new[] { new Dictionary<string, int>(), new Dictionary<string, int>() };
        var counts = new int[EntailProgram.KvGroups];
        for (int group = 0; group < EntailProgram.KvGroups; group++)
        {
            for (int i = 0; i < Pairs; i++)
            {
                layer.K[(group * EntailProgram.HeadDim + i) * h + g.Ballast] = 1f;
            }
        }
        foreach (var head in heads)
        {
            int group = counts[0] < HeadsPerGroup ? 0 : 1;
            int qHead = group * HeadsPerGroup + counts[group]++;
            int block = qHead * 2 * EntailProgram.HeadDim;
            for (int d = 0; d < EntailProgram.HeadDim; d++)
            {
                layer.Q[(block + EntailProgram.HeadDim + d) * h + g.Ballast] = (float)(OutputGate / Math.Pow(2, g.Shift));
            }
            if (head.Pointer is { } pointer)
            {
                foreach (var (lane, offset) in pointer)
                {
                    int dim = g.Lanes.Dim[lane];
                    if (dim < 0)
                    {
                        continue;
                    }
                    var q = OffsetQuery(offset);
                    for (int d = 0; d < EntailProgram.HeadDim; d++)
                    {
                        layer.Q[(block + d) * h + dim] += (float)q[d];
                    }
                }
            }
            else
            {
                var q = OffsetQuery(head.Offset);
                for (int d = 0; d < EntailProgram.HeadDim; d++)
                {
                    layer.Q[(block + d) * h + g.Ballast] += (float)q[d];
                }
            }
            foreach (var (dest, source) in head.Copies)
            {
                if (g.Lanes.Dim[dest] < 0)
                {
                    continue;
                }
                var key = source.Key();
                if (!slots[group].TryGetValue(key, out int slot))
                {
                    slot = slots[group].Count;
                    if (slot >= EntailProgram.HeadDim)
                    {
                        throw new EntailException($"layer {index}: more than {EntailProgram.HeadDim} values in one attention group");
                    }
                    slots[group][key] = slot;
                    ReadInto(layer.V, group * EntailProgram.HeadDim + slot, h, source, g);
                }
                layer.O[g.Lanes.Dim[dest] * QDim + qHead * EntailProgram.HeadDim + slot] = (float)g.Scale[dest];
            }
        }

        // Neurons.
        int at = 0;
        void Place(Lin gate, Lin up, int destDim, double raw)
        {
            if (at >= inter)
            {
                throw new EntailException($"layer {index}: more than {inter} neurons, once dead lanes are cleared; lower --neurons-per-layer to leave room");
            }
            ReadInto(layer.Gate, at, h, gate, g);
            ReadInto(layer.Up, at, h, up, g);
            layer.Down[destDim * inter + at] = (float)raw;
            at++;
        }
        foreach (var n in p.Ops.OfType<Neuron>().Where(x => x.Layer == index))
        {
            int dim = g.Lanes.Dim[n.Dest];
            if (dim >= 0)
            {
                Place(n.Gate, n.Up, dim, n.Coef * g.Scale[n.Dest]);
            }
        }
        foreach (var (gate, up, dest, coef) in extra)
        {
            Place(gate, up, dest, coef);
        }
        foreach (var (gateConst, lane, clearBallast) in clears)
        {
            var gate = Lin.Of(gateConst);
            if (clearBallast)
            {
                for (int d = 0; d < g.BallastDims; d++)
                {
                    Place(gate, Lin.Of(1 / gateConst), g.Ballast + d, -Ballast);
                }
            }
            else
            {
                // Up reads the lane's meaning; the down weight restores raw.
                Place(gate, Lin.OfLane(lane).Scaled(1 / gateConst), g.Lanes.Dim[lane], -g.Scale[lane]);
            }
        }
        return layer;
    }

    private static IEnumerable<TensorPayload> LayerTensors(Layer l, string prefix, int h, int inter)
    {
        yield return F32($"{prefix}.input_layernorm.weight", l.InNorm, h);
        yield return F32($"{prefix}.post_attention_layernorm.weight", l.PostNorm, h);
        yield return F32($"{prefix}.self_attn.q_proj.weight", l.Q, 2 * QDim, h);
        yield return F32($"{prefix}.self_attn.k_proj.weight", l.K, KvDim, h);
        yield return F32($"{prefix}.self_attn.v_proj.weight", l.V, KvDim, h);
        yield return F32($"{prefix}.self_attn.o_proj.weight", l.O, h, QDim);
        yield return F32($"{prefix}.self_attn.q_norm.weight", l.QNorm, EntailProgram.HeadDim);
        yield return F32($"{prefix}.self_attn.k_norm.weight", l.KNorm, EntailProgram.HeadDim);
        yield return F32($"{prefix}.mlp.gate_proj.weight", l.Gate, inter, h);
        yield return F32($"{prefix}.mlp.up_proj.weight", l.Up, inter, h);
        yield return F32($"{prefix}.mlp.down_proj.weight", l.Down, h, inter);
    }

    private static TensorPayload F32(string name, float[] data, params long[] shape)
    {
        var bytes = new byte[data.Length * 4];
        Buffer.BlockCopy(data, 0, bytes, 0, bytes.Length);
        return new TensorPayload { Name = name, Dtype = Dtype.F32, Shape = shape, Data = bytes };
    }

    private static (Lin, Lin, int, double)[] NoExtra => Array.Empty<(Lin, Lin, int, double)>();

    private static IEnumerable<(double, int, bool)> ClearsFor(Allocation a, int layer) =>
        a.Clears.Where(c => c.Layer == layer).Select(c => (EntailCompiler.G, c.Lane, false));

    // ── target 1: a new, empty model ────────────────────────────────────────

    /// <summary>
    /// Writes a complete checkpoint holding nothing but the program: a
    /// byte-level tokenizer over printable ASCII, the compiled layers, and an
    /// output head with one row per <c>emit</c>. No weight is initialised
    /// randomly and none is trained.
    /// </summary>
    public static EntailBuildReport WriteFresh(EntailProgram p, string outDir, int context = 8192)
    {
        RefuseVotes(p);
        if (p.Masks.Count > 0)
        {
            throw new EntailException("`only` and `forbid` narrow a host's predictions: graft this program instead");
        }
        // Vocabulary: two control tokens, then printable ASCII.
        var vocab = new List<string> { "<none>", "<eos>" };
        for (char c = ' '; c <= '~'; c++)
        {
            vocab.Add(c.ToString());
        }
        var ids = vocab.Select((t, i) => (t, i)).ToDictionary(x => x.t, x => x.i, StringComparer.Ordinal);
        var missing = p.Tokens.Keys.Where(t => !ids.ContainsKey(t)).ToList();
        var inputs = p.Tokens.Keys.Where(ids.ContainsKey).Select(t => (ids[t], t)).ToList();
        var outputs = p.Emits.Select(e => (e.Text is null ? 1 : ids[e.Text], e.Lane, 64f)).ToList();
        return WriteBlank(p, outDir, context, vocab, inputs, outputs, missing);
    }

    /// <summary>
    /// Writes a program as an adapter to run beside a partner model: the same
    /// blank model as <see cref="WriteFresh"/>, reading the partner's token ids
    /// through a table, and with one output row per output lane. Its logits are
    /// therefore its verdict, and all zero where it has nothing to say.
    /// </summary>
    public static (EntailBuildReport Report, EntailAdapterManifest Manifest) WriteAdapter(
        EntailProgram p, string partnerTokenizerDir, string outDir, int context = 8192)
    {
        var binding = BindAdapter(p, partnerTokenizerDir);
        int size = Math.Max(binding.Manifest.Rows.Count + 2, binding.Manifest.Outputs.Count);
        var vocab = new List<string> { "<none>", "<eos>" };
        vocab.AddRange(Enumerable.Range(2, size - 2).Select(i => $"r{i}"));
        var outputs = binding.Outputs.Select(o => (o.Row, o.Lane, 1f)).ToList();
        var report = WriteBlank(p, outDir, context, vocab, binding.Inputs, outputs, binding.Missing);
        binding.Manifest.Save(outDir);
        return (report, binding.Manifest);
    }

    /// <summary>Joins a program to a partner's tokens: which token each input
    /// row is, and what each output row means.</summary>
    public static EntailAdapterBinding BindAdapter(EntailProgram p, string partnerTokenizerDir)
    {
        RefuseVotes(p);
        var (ids, missing) = BindTokens(p, partnerTokenizerDir);

        // Row 0 stands for every token the program does not know, row 1 is unused.
        var rows = new Dictionary<int, int>();
        var inputs = new List<(int Row, string Text)>();
        foreach (var text in p.Tokens.Keys.Where(ids.ContainsKey))
        {
            int row = rows.Count + 2;
            rows[ids[text]] = row;
            inputs.Add((row, text));
        }

        var bound = p.Emits.Where(e => e.Text is not null && ids.ContainsKey(e.Text)).ToList();
        var said = bound.Where(e => e.Text != " ").Select(e => ids[e.Text!]).Distinct().ToArray();
        var entries = new List<EntailAdapterOutput>();
        var outputs = new List<(int Row, int Lane)>();
        void Add(EntailOutputKind kind, int lane, int[] tokens)
        {
            outputs.Add((entries.Count, lane));
            entries.Add(new EntailAdapterOutput(entries.Count, kind, tokens));
        }
        foreach (var (text, lane) in p.Emits)
        {
            if (text is null)
            {
                Add(EntailOutputKind.End, lane, said);
            }
            else if (ids.TryGetValue(text, out int id))
            {
                Add(EntailOutputKind.Say, lane, new[] { id });
            }
        }
        foreach (var (allow, texts, lane) in p.Masks)
        {
            Add(allow ? EntailOutputKind.Only : EntailOutputKind.Forbid, lane,
                texts.Where(ids.ContainsKey).Select(t => ids[t]).Distinct().ToArray());
        }

        var manifest = new EntailAdapterManifest(p.Name,
            EntailAdapterManifest.TokenizerHash(partnerTokenizerDir), rows, entries);
        return new EntailAdapterBinding(manifest, inputs, outputs, missing);
    }

    private static EntailBuildReport WriteBlank(EntailProgram p, string outDir, int context, IReadOnlyList<string> vocab,
        IReadOnlyList<(int Row, string Text)> inputs, IReadOnlyList<(int Row, int Lane, float Weight)> outputs, List<string> missing)
    {
        Directory.CreateDirectory(outDir);
        int hidden = WidthFor(Allocate(p, firstDim: 0, keepEmitsLive: true).Used);
        var (ballastDims, shift) = BallastShape(hidden);
        var alloc = Allocate(p, firstDim: ballastDims, keepEmitsLive: true);
        int clearPeak = Enumerable.Range(0, p.Layers).Max(l =>
            p.Ops.OfType<Neuron>().Count(n => n.Layer == l) + alloc.Clears.Count(c => c.Layer == l));
        int intermediate = RoundUp(clearPeak, 64);
        var scale = p.Lanes.Select(l => l.Scale).ToArray();
        var geometry = new Geometry
        {
            Hidden = hidden, Intermediate = intermediate, Ballast = 0, BallastDims = ballastDims, Shift = shift, Lanes = alloc,
            Scale = scale, HostDim = new bool[hidden],
        };

        var embed = new float[(long)vocab.Count * hidden];
        for (int t = 0; t < vocab.Count; t++)
        {
            for (int d = 0; d < ballastDims; d++)
            {
                embed[(long)t * hidden + d] = (float)Ballast;
            }
        }
        foreach (var (row, text) in inputs)
        {
            foreach (var (lane, value) in p.Tokens[text])
            {
                if (alloc.Dim[lane] >= 0)
                {
                    embed[(long)row * hidden + alloc.Dim[lane]] = (float)(value * scale[lane]);
                }
            }
        }

        var head = new float[(long)vocab.Count * hidden];
        foreach (var (row, lane, weight) in outputs)
        {
            if (alloc.Dim[lane] >= 0)
            {
                head[(long)row * hidden + alloc.Dim[lane]] = weight;
            }
        }

        var tensors = new List<TensorPayload> { F32("model.embed_tokens.weight", embed, vocab.Count, hidden) };
        for (int l = 0; l < p.Layers; l++)
        {
            var layer = BuildLayer(p, geometry, l, NoExtra, ClearsFor(alloc, l));
            tensors.AddRange(LayerTensors(layer, $"model.layers.{l}", hidden, intermediate));
        }
        tensors.Add(F32("model.norm.weight", NormRow(geometry, Ballast / Math.Pow(2, shift)), hidden));
        tensors.Add(F32("lm_head.weight", head, vocab.Count, hidden));
        SafetensorsWriter.Write(Path.Combine(outDir, "model.safetensors"), tensors);

        var config = new JsonObject
        {
            ["architectures"] = new JsonArray("Qwen3_5ForConditionalGeneration"),
            ["model_type"] = "qwen3_5_text",
            ["hidden_size"] = hidden,
            ["num_hidden_layers"] = p.Layers,
            ["num_attention_heads"] = EntailProgram.HeadsPerLayer,
            ["num_key_value_heads"] = EntailProgram.KvGroups,
            ["head_dim"] = EntailProgram.HeadDim,
            ["intermediate_size"] = intermediate,
            ["hidden_act"] = "silu",
            ["rms_norm_eps"] = 1e-6,
            ["vocab_size"] = vocab.Count,
            ["max_position_embeddings"] = context,
            ["layer_types"] = new JsonArray(Enumerable.Repeat("full_attention", p.Layers).Select(x => (JsonNode?)x).ToArray()),
            ["attn_output_gate"] = true,
            ["tie_word_embeddings"] = false,
            ["eos_token_id"] = 1,
            ["rope_parameters"] = new JsonObject
            {
                ["rope_type"] = "default",
                ["rope_theta"] = RopeTheta,
                ["partial_rotary_factor"] = 0.25,
            },
        };
        File.WriteAllText(Path.Combine(outDir, "config.json"), config.ToJsonString(new JsonSerializerOptions { WriteIndented = true }));
        File.WriteAllText(Path.Combine(outDir, "tokenizer.json"), TokenizerJson(vocab));
        File.WriteAllText(Path.Combine(outDir, "generation_config.json"), "{ \"eos_token_id\": 1 }");

        return new EntailBuildReport(outDir, hidden, p.Layers, p.Layers, alloc.Used, alloc.Peak,
            p.Ops.Count(o => o is Neuron), p.Ops.Count(o => o is Head), AttentionGap(p, context), inputs.Count, missing);
    }

    private static string TokenizerJson(IReadOnlyList<string> vocab)
    {
        var map = new JsonObject();
        for (int i = 0; i < vocab.Count; i++)
        {
            map[vocab[i] == " " ? "Ġ" : vocab[i]] = i;
        }
        var root = new JsonObject
        {
            ["version"] = "1.0.0",
            ["truncation"] = null,
            ["padding"] = null,
            ["added_tokens"] = new JsonArray(
                new JsonObject { ["id"] = 0, ["content"] = "<none>", ["single_word"] = false, ["lstrip"] = false, ["rstrip"] = false, ["normalized"] = false, ["special"] = true },
                new JsonObject { ["id"] = 1, ["content"] = "<eos>", ["single_word"] = false, ["lstrip"] = false, ["rstrip"] = false, ["normalized"] = false, ["special"] = true }),
            ["normalizer"] = null,
            ["pre_tokenizer"] = new JsonObject
            {
                ["type"] = "Sequence",
                ["pretokenizers"] = new JsonArray(
                    new JsonObject
                    {
                        ["type"] = "Split",
                        ["pattern"] = new JsonObject { ["Regex"] = "[\\s\\S]" },
                        ["behavior"] = "Isolated",
                        ["invert"] = false,
                    },
                    new JsonObject { ["type"] = "ByteLevel", ["add_prefix_space"] = false, ["trim_offsets"] = false, ["use_regex"] = false }),
            },
            ["post_processor"] = null,
            ["decoder"] = new JsonObject { ["type"] = "ByteLevel", ["add_prefix_space"] = false, ["trim_offsets"] = false, ["use_regex"] = false },
            ["model"] = new JsonObject
            {
                ["type"] = "BPE",
                ["dropout"] = null,
                ["unk_token"] = null,
                ["continuing_subword_prefix"] = null,
                ["end_of_word_suffix"] = null,
                ["fuse_unk"] = false,
                ["byte_fallback"] = false,
                ["ignore_merges"] = false,
                ["vocab"] = map,
                ["merges"] = new JsonArray(),
            },
        };
        return root.ToJsonString();
    }

    private static void RefuseVotes(EntailProgram p)
    {
        if (p.Votes.Count > 0)
        {
            throw new EntailException("this program still has soft votes: refine them into `emit` before compiling it");
        }
    }

    private static int RoundUp(int value, int unit) => (value + unit - 1) / unit * unit;

    // ── target 2: grafted into a host checkpoint, in situ ───────────────────

    /// <summary>Raw height of a set output lane: small enough that the host's
    /// final norm cannot feel it.</summary>
    private const double EmitLaneHeight = 1.0 / 1024;
    /// <summary>The token's own entry in its output lane. With a tied head this
    /// column is both an input and an output, so the block clears it on the
    /// way in.</summary>
    private const double EmitRow = 1.0 / 1048576;
    /// <summary>Residual norm the host reaches at its final norm, and how
    /// many logits a set output lane adds at that norm.</summary>
    private const double HostOutputNorm = 115;
    private const double OutputForce = 256;

    /// <summary>
    /// Restructures a Qwen3.5-family text checkpoint around the program,
    /// leaving every host weight's value in place:
    /// <list type="bullet">
    /// <item>The hidden size widens; the new dimensions are the program's
    /// lanes. Every host matrix gains zeros there, so no host weight can read
    /// or write a lane, and its norm weights are rescaled so the wider mean
    /// leaves them exactly where they were.</item>
    /// <item>The compiled layers go in front of the host's first layer. They
    /// read only lanes, so they compute a pure function of the tokens. They
    /// finish by clearing every lane and the ballast: the host then starts
    /// from its own unchanged embedding.</item>
    /// <item>Each output lane is a column of the tied embedding, non-zero in
    /// its own token's row alone, and the only lanes the final norm passes.
    /// A set lane therefore raises exactly one logit; the end-of-answer lane
    /// lowers the logits of the tokens an answer is made of.</item>
    /// </list>
    /// Wherever the program says nothing, the host's computation is unchanged.
    /// </summary>
    public static EntailBuildReport WriteGraft(EntailProgram p, string hostDir, string outDir, int context = 8192)
    {
        RefuseVotes(p);
        Directory.CreateDirectory(outDir);
        var hostConfig = JsonNode.Parse(File.ReadAllText(Path.Combine(hostDir, "config.json")))!.AsObject();
        var text = hostConfig["text_config"]?.AsObject() ?? hostConfig;
        int hostHidden = text["hidden_size"]!.GetValue<int>();
        int hostLayers = text["num_hidden_layers"]!.GetValue<int>();
        int inter = text["intermediate_size"]!.GetValue<int>();
        int headDim = text["head_dim"]!.GetValue<int>();
        if (headDim != EntailProgram.HeadDim ||
            text["num_attention_heads"]!.GetValue<int>() != EntailProgram.HeadsPerLayer ||
            text["num_key_value_heads"]!.GetValue<int>() != EntailProgram.KvGroups)
        {
            throw new EntailException("the graft target must have 8 query heads, 2 KV heads and head_dim 256");
        }

        int hidden = WidthFor(hostHidden + Allocate(p, firstDim: 0, keepEmitsLive: true).Used);
        var (ballastDims, shift) = BallastShape(hidden);
        var alloc = Allocate(p, firstDim: hostHidden + ballastDims, keepEmitsLive: true);
        int ballast = hostHidden;
        var scale = p.Lanes.Select(l => l.Emits ? EmitLaneHeight : l.Scale).ToArray();
        var hostDim = new bool[hidden];
        for (int d = 0; d < hostHidden; d++)
        {
            hostDim[d] = true;
        }
        var geometry = new Geometry
        {
            Hidden = hidden, Intermediate = inter, Ballast = ballast, BallastDims = ballastDims, Shift = shift, Lanes = alloc,
            Scale = scale, HostDim = hostDim,
        };

        using var host = SafetensorsFile.Open(Path.Combine(hostDir, "model.safetensors"));
        string Prefix(string name) => name.StartsWith("model.language_model.", StringComparison.Ordinal)
            ? "model.language_model." : "model.";
        string root = Prefix(host.TensorNames.First(n => n.Contains("embed_tokens", StringComparison.Ordinal)));
        string embedName = root + "embed_tokens.weight";

        var tensors = new List<TensorPayload>();
        var (vocabIds, missing) = BindTokens(p, hostDir);

        // Embedding: host dims as they are, then ballast and token lanes.
        var embedInfo = host.GetTensor(embedName);
        if (embedInfo.Dtype != Dtype.BF16)
        {
            throw new EntailException($"expected a bf16 host embedding, found {embedInfo.Dtype.Label()}");
        }
        long vocab = embedInfo.Shape[0];
        var hostEmbed = host.ReadBytes(embedInfo);
        var embed = new byte[vocab * hidden * 2];
        ushort ballastBits = Bf16(Ballast);
        for (long t = 0; t < vocab; t++)
        {
            Buffer.BlockCopy(hostEmbed, (int)(t * hostHidden * 2), embed, (int)(t * hidden * 2), hostHidden * 2);
            for (int d = 0; d < ballastDims; d++)
            {
                WriteU16(embed, t * hidden + ballast + d, ballastBits);
            }
        }
        foreach (var (textToken, id) in vocabIds)
        {
            // A token that is only ever said has no input facts.
            if (!p.Tokens.TryGetValue(textToken, out var writes))
            {
                continue;
            }
            foreach (var (lane, value) in writes)
            {
                if (alloc.Dim[lane] >= 0)
                {
                    WriteU16(embed, (long)id * hidden + alloc.Dim[lane], Bf16(value * scale[lane]));
                }
            }
        }
        var outputs = p.Emits.Where(e => e.Text is not null && vocabIds.ContainsKey(e.Text)).ToList();
        foreach (var (token, lane) in outputs)
        {
            WriteU16(embed, (long)vocabIds[token!] * hidden + alloc.Dim[lane], Bf16(EmitRow));
        }
        // "End of answer" has no token in a host, whose text simply goes on.
        // Its lane is a column too, negative in the rows of the tokens an
        // answer is made of: the number stops, and the host says what follows.
        var ends = p.Emits.Where(e => e.Text is null).Select(e => e.Lane).ToList();
        foreach (int lane in ends)
        {
            foreach (var (token, _) in outputs.Where(o => o.Text != " "))
            {
                WriteU16(embed, (long)vocabIds[token!] * hidden + alloc.Dim[lane], Bf16(-EmitRow));
            }
        }
        // A mask's lane is a column too, negative in the rows it rules out.
        foreach (var (allow, texts, lane) in p.Masks)
        {
            var listed = texts.Where(vocabIds.ContainsKey).Select(t => vocabIds[t]).ToHashSet();
            for (int id = 0; id < vocab; id++)
            {
                if (listed.Contains(id) != allow)
                {
                    WriteU16(embed, (long)id * hidden + alloc.Dim[lane], Bf16(-EmitRow));
                }
            }
        }
        var outputLanes = outputs.Select(o => o.Lane).Concat(ends).Concat(p.Masks.Select(m => m.Lane)).ToList();
        tensors.Add(new TensorPayload { Name = embedName, Dtype = Dtype.BF16, Shape = new[] { vocab, hidden }, Data = embed });

        // The compiled block, then the two layers that hand over to the host.
        int block = p.Layers + 2;
        for (int l = 0; l < p.Layers; l++)
        {
            var clears = ClearsFor(alloc, l);
            if (l == 0)
            {
                clears = clears.Concat(outputLanes.Select(lane => (EntailCompiler.G, lane, false)));
            }
            var layer = BuildLayer(p, geometry, l, NoExtra, clears);
            tensors.AddRange(LayerTensors(layer, $"{root}layers.{l}", hidden, inter));
        }
        var finalClears = alloc.LiveAtEnd.Select(lane => (EntailCompiler.G, lane, false));
        tensors.AddRange(LayerTensors(BuildLayer(p, geometry, -1, NoExtra, finalClears),
            $"{root}layers.{p.Layers}", hidden, inter));
        tensors.AddRange(LayerTensors(BuildLayer(p, geometry, -1, NoExtra,
            new[] { (EntailCompiler.G, 0, true) }), $"{root}layers.{p.Layers + 1}", hidden, inter));

        // The host's own tensors, widened.
        double shrink = Math.Sqrt(hostHidden / (double)hidden);
        foreach (var name in host.TensorNames)
        {
            if (!name.StartsWith(root, StringComparison.Ordinal) || name == embedName)
            {
                continue;
            }
            var info = host.GetTensor(name);
            var relative = name[root.Length..];
            string target = name;
            if (relative.StartsWith("layers.", StringComparison.Ordinal))
            {
                int dot = relative.IndexOf('.', 7);
                int index = int.Parse(relative[7..dot]);
                target = $"{root}layers.{index + block}{relative[dot..]}";
            }
            var widened = Widen(host, info, target, hostHidden, hidden, shrink, isFinalNorm: relative == "norm.weight");
            if (relative == "norm.weight")
            {
                // logit += EmitRow · gamma · lane/rms: OutputForce at the host's usual norm.
                double gamma = OutputForce * HostOutputNorm / (EmitRow * EmitLaneHeight * Math.Sqrt(hidden));
                foreach (int lane in outputLanes)
                {
                    BitConverter.TryWriteBytes(widened.Data.AsSpan(alloc.Dim[lane] * 4), (float)(gamma - 1));
                }
            }
            tensors.Add(widened);
        }

        SafetensorsWriter.Write(Path.Combine(outDir, "model.safetensors"), tensors);

        var types = new JsonArray();
        for (int i = 0; i < block; i++)
        {
            types.Add("full_attention");
        }
        foreach (var t in text["layer_types"]!.AsArray())
        {
            types.Add(t!.GetValue<string>());
        }
        var config = JsonNode.Parse(hostConfig.ToJsonString())!.AsObject();
        var target2 = config["text_config"]?.AsObject() ?? config;
        target2["hidden_size"] = hidden;
        target2["num_hidden_layers"] = types.Count;
        target2["layer_types"] = types;
        // rms_norm_eps stays as it is: the same value serves the per-head
        // norms of linear attention, which do not widen, and they are far more
        // sensitive to it than the layer norms are.
        config.Remove("vision_config");
        File.WriteAllText(Path.Combine(outDir, "config.json"), config.ToJsonString(new JsonSerializerOptions { WriteIndented = true }));
        foreach (var file in new[] { "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json" })
        {
            var source = Path.Combine(hostDir, file);
            if (File.Exists(source))
            {
                File.Copy(source, Path.Combine(outDir, file), overwrite: true);
            }
        }

        return new EntailBuildReport(outDir, hidden, types.Count, block, alloc.Used, alloc.Peak,
            p.Ops.Count(o => o is Neuron), p.Ops.Count(o => o is Head), AttentionGap(p, context),
            vocabIds.Count, missing);
    }

    private static TensorPayload Widen(SafetensorsFile host, TensorInfo info, string target,
        int hostHidden, int hidden, double shrink, bool isFinalNorm)
    {
        var bytes = host.ReadBytes(info);
        int element = info.Dtype == Dtype.BF16 ? 2 : 4;
        bool norm = info.Shape.Length == 1 && info.Shape[0] == hostHidden;
        if (norm)
        {
            // (1+w)·shrink keeps the host's normalised values where they were
            // under a mean over more dimensions. The final norm additionally
            // zeroes the lanes, so they never reach the output head.
            var widened = BitPattern.WidenToF32(info.Dtype, bytes);
            var w = new float[hidden];
            for (int d = 0; d < hidden; d++)
            {
                w[d] = d < hostHidden ? (float)((1 + (double)widened[d]) * shrink - 1) : isFinalNorm ? -1f : 0f;
            }
            return F32(target, w, hidden);
        }
        if (info.Shape.Length == 2 && info.Shape[1] == hostHidden)
        {
            long rows = info.Shape[0];
            var data = new byte[rows * hidden * element];
            for (long r = 0; r < rows; r++)
            {
                Buffer.BlockCopy(bytes, (int)(r * hostHidden * element), data, (int)(r * hidden * element), hostHidden * element);
            }
            return new TensorPayload { Name = target, Dtype = info.Dtype, Shape = new[] { rows, (long)hidden }, Data = data };
        }
        if (info.Shape.Length == 2 && info.Shape[0] == hostHidden)
        {
            var data = new byte[hidden * info.Shape[1] * element];
            Buffer.BlockCopy(bytes, 0, data, 0, bytes.Length);
            return new TensorPayload { Name = target, Dtype = info.Dtype, Shape = new[] { (long)hidden, info.Shape[1] }, Data = data };
        }
        return new TensorPayload { Name = target, Dtype = info.Dtype, Shape = info.Shape, Data = bytes };
    }

    private static void WriteU16(byte[] data, long element, ushort value)
    {
        data[element * 2] = (byte)value;
        data[element * 2 + 1] = (byte)(value >> 8);
    }

    private static ushort Bf16(double value)
    {
        uint bits = BitConverter.SingleToUInt32Bits((float)value);
        return (ushort)((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16);
    }

    /// <summary>Finds each program token in a byte-level BPE vocabulary. A
    /// token the vocabulary does not hold as one piece is simply not part of
    /// this model's alphabet.</summary>
    private static (Dictionary<string, int> Ids, List<string> Missing) BindTokens(EntailProgram p, string hostDir)
    {
        var root = JsonNode.Parse(File.ReadAllText(Path.Combine(hostDir, "tokenizer.json")))!.AsObject();
        var vocab = root["model"]!["vocab"]!.AsObject();
        var byteMap = ByteLevelMap();
        var added = (root["added_tokens"]?.AsArray() ?? new JsonArray()).ToDictionary(
            t => t!["content"]!.GetValue<string>(), t => t!["id"]!.GetValue<int>(), StringComparer.Ordinal);
        var ids = new Dictionary<string, int>(StringComparer.Ordinal);
        var missing = new List<string>();
        foreach (var token in p.Tokens.Keys.Concat(p.Emits.Where(e => e.Text is not null).Select(e => e.Text!))
            .Concat(p.Masks.SelectMany(m => m.Texts)).Distinct())
        {
            var encoded = string.Concat(Encoding.UTF8.GetBytes(token).Select(b => byteMap[b]));
            if (vocab.TryGetPropertyValue(encoded, out var id) && id is not null)
            {
                ids[token] = id.GetValue<int>();
            }
            else if (added.TryGetValue(token, out int special))
            {
                ids[token] = special;
            }
            else
            {
                missing.Add(token);
            }
        }
        return (ids, missing);
    }

    private static char[] ByteLevelMap()
    {
        var map = new char[256];
        int extra = 0;
        for (int b = 0; b < 256; b++)
        {
            bool printable = b is >= 33 and <= 126 or >= 161 and <= 172 or >= 174 and <= 255;
            map[b] = printable ? (char)b : (char)(256 + extra++);
        }
        return map;
    }
}
