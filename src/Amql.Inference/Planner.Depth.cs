using Amql.Vindex3;

namespace Amql.Inference;

/// <summary>
/// The planner's depth and conditioning ops: a depth loop's visit schedule, the
/// block residual's mixes, message gates, residual short convolutions, the
/// attention's conditioning (separate inputs, query scale and clip, position
/// log-scaling, relative bias, sinks, a separate output gate) and packed
/// experts. Each is bound from its surface and the layer role map; none names a
/// model family.
/// </summary>
public static partial class Planner
{
    /// <summary>A layer plan with the depth features of its surface bound: the
    /// attention's separate input norms, the block residual's mixes, message
    /// gates on looped layers, and residual convolutions on the sub-block outputs.</summary>
    private static LayerPlan DecorateDepth(OperandStore store, string stackId, int layer, LayerPlan plan, ExecutionSurface surface,
        NormSurface normSurface, LayerRoleMap roles, int hidden, int numLayers)
    {
        NormOp? qIn = null, kIn = null, vIn = null;
        if (plan.Attention is { SeparateInputs: true })
        {
            qIn = RequiredNorm(TensorRoles.QInNorm);
            kIn = RequiredNorm(TensorRoles.KInNorm);
            vIn = RequiredNorm(TensorRoles.VInNorm);
            if (plan.PreAttentionNorm is not null)
            {
                throw new ContainerException($"layer {layer}: separate q, k and v input norms beside a single attention input norm");
            }
        }
        else if (RoleTensor(store, stackId, layer, roles, TensorRoles.KInNorm) is not null ||
                 RoleTensor(store, stackId, layer, roles, TensorRoles.VInNorm) is not null)
        {
            throw new ContainerException($"layer {layer}: k or v input norms without a q input norm");
        }

        DepthMixOp? mixerMix = null, ffnMix = null;
        bool opensBlock = false;
        if (surface.BlockResidual is { } block)
        {
            if (block.MixerQueries.Count != numLayers || block.FfnQueries.Count != numLayers || block.BlockSize <= 0)
            {
                throw new ContainerException(
                    $"block residual: {block.MixerQueries.Count} mixer and {block.FfnQueries.Count} FFN queries for {numLayers} layers in blocks of {block.BlockSize}");
            }
            if (block.MixerQueries[layer] is { } mixerQuery)
            {
                mixerMix = BindMix(store, block.QueryObject, mixerQuery, block.Eps, hidden, layer);
                if (mixerMix.Count is not (1 or 3) || (mixerMix.Count == 3 && plan.Attention is not { SeparateInputs: true }))
                {
                    throw new ContainerException(
                        $"layer {layer}: a mixer input mix of {mixerMix.Count} queries (one, or three for an attention with separate q, k and v inputs)");
                }
            }
            ffnMix = BindMix(store, block.QueryObject, block.FfnQueries[layer], block.Eps, hidden, layer);
            if (ffnMix.Count != 1)
            {
                throw new ContainerException($"layer {layer}: the FFN's input mix has {ffnMix.Count} queries, not one");
            }
            opensBlock = layer % block.BlockSize == 0;
            if (plan.Attention is null)
            {
                throw new UnsupportedOperatorException($"layer {layer}: a block residual around a non-softmax mixer has no managed implementation");
            }
        }

        List<MessageGateOp>? gates = null;
        if (surface.MessageGates is { } messages)
        {
            var loop = surface.DepthLoop ?? throw new ContainerException("message gates without a depth loop to read messages from");
            var bankSurface = surface.BlockResidual ?? throw new ContainerException("message gates read a block residual's bank, and the stack has none");
            if (layer >= loop.Start && layer < loop.End)
            {
                gates = new List<MessageGateOp>();
                foreach (var site in messages.Sites)
                {
                    if (site.Input is not ("q" or "k" or "v" or "ffn"))
                    {
                        throw new ContainerException($"message gate site '{site.Input}' is not q, k, v or ffn");
                    }
                    if (site.Input != "ffn" && plan.Attention is null)
                    {
                        throw new UnsupportedOperatorException($"layer {layer}: message gates on a non-softmax mixer have no managed implementation");
                    }
                    string role = TensorRoles.MessageGate(site.Input);
                    OperandRef Part(string part, params long[] shape)
                    {
                        var op = RoleOperand(store, stackId, layer, roles, role, part)
                                 ?? throw new ContainerException($"operand closure: layer {layer}: message gate '{site.Input}' has no '{part}'");
                        CheckShape(store, op, layer, $"message gate {site.Input} {part}", shape);
                        return op;
                    }
                    int row = site.Offset + site.Stride * (layer - loop.Start);
                    var query = new OperandRef(bankSurface.QueryObject, site.Query);
                    Require(store, query.ObjectId, query.TensorName, layer, $"message query ({site.Input})");
                    var queryShape = store.ShapeOf(query.ObjectId, query.TensorName);
                    if (queryShape.Length != 2 || queryShape[1] != hidden || row < 0 || row >= queryShape[0])
                    {
                        throw new ContainerException(
                            $"layer {layer}: message query '{site.Query}' [{string.Join(", ", queryShape)}] has no row {row} of width {hidden}");
                    }
                    gates.Add(new MessageGateOp
                    {
                        Input = site.Input,
                        Down = Part("down", messages.Rank, 2L * hidden),
                        Forget = Part("forget", hidden, messages.Rank),
                        Gain = Part("gain", hidden, messages.Rank),
                        Query = query,
                        QueryRow = row,
                        Rank = messages.Rank,
                        Eps = messages.Eps,
                        MixEps = bankSurface.Eps,
                    });
                }
            }
        }

        ResidualConvOp? mixerConv = null, ffnConv = null;
        if (surface.ResidualConv is { } conv)
        {
            mixerConv = BindStaticConv(store, stackId, layer, roles, TensorRoles.MixerOutConv, conv.Kernel, hidden);
            ffnConv = BindStaticConv(store, stackId, layer, roles, TensorRoles.FfnOutConv, conv.Kernel, hidden);
        }

        if (qIn is null && mixerMix is null && ffnMix is null && !opensBlock && gates is null && mixerConv is null && ffnConv is null)
        {
            return plan;
        }
        return new LayerPlan
        {
            Attention = plan.Attention,
            LinearAttention = plan.LinearAttention,
            Conv = plan.Conv,
            PreAttentionNorm = plan.PreAttentionNorm,
            PostAttentionNorm = plan.PostAttentionNorm,
            PreFfnNorm = plan.PreFfnNorm,
            PostFfnNorm = plan.PostFfnNorm,
            Ffn = plan.Ffn,
            MixerHyperConnection = plan.MixerHyperConnection,
            FfnHyperConnection = plan.FfnHyperConnection,
            Ngram = plan.Ngram,
            OutputScale = plan.OutputScale,
            ParallelResidual = plan.ParallelResidual,
            QInputNorm = qIn,
            KInputNorm = kIn,
            VInputNorm = vIn,
            MixerInputMix = mixerMix,
            FfnInputMix = ffnMix,
            OpensBlock = opensBlock,
            MessageGates = gates,
            MixerOutputConv = mixerConv,
            FfnOutputConv = ffnConv,
        };

        NormOp RequiredNorm(string role) =>
            BindRoleNorm(store, stackId, layer, roles, role, normSurface.Pre, hidden)
            ?? throw new ContainerException($"operand closure: layer {layer}: separate attention inputs, but no '{role}' norm");
    }

    private static DepthMixOp BindMix(OperandStore store, string objectId, string tensor, double eps, int hidden, int layer)
    {
        Require(store, objectId, tensor, layer, "depth-mix query");
        var shape = store.ShapeOf(objectId, tensor);
        int count = shape switch
        {
            [var w] when w == hidden => 1,
            [var n, var w] when w == hidden => checked((int)n),
            _ => throw new ContainerException(
                $"operand closure: '{objectId}/{tensor}' [{string.Join(", ", shape)}] is not a depth-mix query of width {hidden}"),
        };
        return new DepthMixOp { Queries = new OperandRef(objectId, tensor), Count = count, Eps = eps };
    }

    /// <summary>The visit schedule of a depth loop: every layer before the loop
    /// once, the loop's layers once per pass (pass 0 in the layer's own cache
    /// slot, later passes in slots past the last layer), then the rest.</summary>
    private static (DepthLoopOp? Loop, IReadOnlyList<LayerVisit>? Visits) DepthSchedule(ExecutionSurface surface, int numLayers,
        List<LayerPlan> layers)
    {
        if (surface.DepthLoop is not { } loop)
        {
            if (surface.MessageGates is not null)
            {
                throw new ContainerException("message gates without a depth loop");
            }
            return (null, null);
        }
        if (loop.Start < 0 || loop.End <= loop.Start || loop.End > numLayers || loop.Passes < 1)
        {
            throw new ContainerException($"depth loop [{loop.Start}, {loop.End}) x{loop.Passes} over {numLayers} layers");
        }
        if (layers.Any(l => l.LinearAttention is not null || l.Conv is not null || l.MixerHyperConnection is not null))
        {
            throw new UnsupportedOperatorException("a depth loop over recurrent, short-conv or gated-residual layers has no managed implementation");
        }
        int core = loop.End - loop.Start;
        var visits = new List<LayerVisit>();
        for (int l = 0; l < loop.Start; l++)
        {
            visits.Add(new LayerVisit(l, 0, l));
        }
        for (int pass = 0; pass < loop.Passes; pass++)
        {
            for (int l = loop.Start; l < loop.End; l++)
            {
                visits.Add(new LayerVisit(l, pass, pass == 0 ? l : numLayers + (pass - 1) * core + (l - loop.Start)));
            }
        }
        for (int l = loop.End; l < numLayers; l++)
        {
            visits.Add(new LayerVisit(l, 0, l));
        }
        return (new DepthLoopOp
        {
            Start = loop.Start,
            End = loop.End,
            Passes = loop.Passes,
            SkipBlocks = surface.MessageGates?.SkipBlocks ?? 0,
        }, visits);
    }

    private static BlockResidualOp? BindBlockResidual(OperandStore store, ExecutionSurface surface, int hidden)
    {
        if (surface.BlockResidual is not { } block)
        {
            return null;
        }
        var output = BindMix(store, block.QueryObject, block.OutputQuery, block.Eps, hidden, -1);
        if (output.Count != 1)
        {
            throw new ContainerException($"the final norm's depth mix has {output.Count} queries, not one");
        }
        return new BlockResidualOp { BlockSize = block.BlockSize, OutputQuery = output };
    }

    // ── attention conditioning ───────────────────────────────────────────────

    private static RelativeBiasOp? BindRelativeBias(OperandStore store, string stackId, int layer, LayerRoleMap roles,
        AttentionSurface attn, long hidden)
    {
        if (attn.RelativeBias is not { } bias)
        {
            return null;
        }
        OperandRef Role(string role) => RoleOperand(store, stackId, layer, roles, role)
            ?? throw new ContainerException($"operand closure: layer {layer}: a relative bias implies '{role}', and the layer has none");
        var projection = Role(TensorRoles.RelProj);
        var scale = Role(TensorRoles.RelChannelScale);
        var table = Role(TensorRoles.RelTable);
        CheckShape(store, projection, layer, "relative-bias projection", (long)attn.NumQHeads * bias.Rank, hidden);
        CheckShape(store, scale, layer, "relative-bias scale", attn.NumQHeads, bias.Rank);
        var tableShape = store.ShapeOf(table.ObjectId, table.TensorName);
        if (tableShape.Length != 2 || tableShape[0] != bias.Rank || tableShape[1] < 1)
        {
            throw new ContainerException($"layer {layer}: relative-bias table [{string.Join(", ", tableShape)}] is not [{bias.Rank}, extent]");
        }
        return new RelativeBiasOp
        {
            Projection = projection,
            ChannelScale = scale,
            Table = table,
            Rank = bias.Rank,
            Extent = checked((int)tableShape[1]),
            Clip = bias.Clip,
        };
    }

    private static AttentionSinkOp? BindSink(OperandStore store, string stackId, int layer, LayerRoleMap roles, AttentionSurface attn)
    {
        if (attn.SinkLogits is not { } kind)
        {
            return null;
        }
        string role = kind == SinkLogitKind.Projected ? TensorRoles.SinkProj : TensorRoles.Sink;
        var weight = RoleOperand(store, stackId, layer, roles, role)
                     ?? throw new ContainerException($"operand closure: layer {layer}: {kind} sink logits imply '{role}', and the layer has none");
        var shape = store.ShapeOf(weight.ObjectId, weight.TensorName);
        bool fits = kind == SinkLogitKind.Projected ? shape is [var h, _] && h == attn.NumQHeads : shape is [var n] && n == attn.NumQHeads;
        if (!fits)
        {
            throw new ContainerException($"layer {layer}: sink '{weight.TensorName}' [{string.Join(", ", shape)}] does not give one logit per head");
        }
        return new AttentionSinkOp { Weight = weight, Projected = kind == SinkLogitKind.Projected };
    }

    /// <summary>The residual convolution on a key or value projection: dynamic
    /// when the role has <c>static</c> and <c>tap_proj</c> parts, static when it has
    /// a <c>weight</c>, none otherwise.</summary>
    private static ResidualConvOp? BindProjectionConv(OperandStore store, string stackId, int layer, LayerRoleMap roles, string role,
        ExecutionSurface surface, HeadGeometry geometry)
    {
        if (surface.ResidualConv is not { } conv)
        {
            return null;
        }
        int channels = geometry.NumKvHeads * geometry.HeadDim;
        var statics = RoleOperand(store, stackId, layer, roles, role, "static");
        var tapProj = RoleOperand(store, stackId, layer, roles, role, "tap_proj");
        if (statics is not null || tapProj is not null)
        {
            if (statics is null || tapProj is null)
            {
                throw new ContainerException($"operand closure: layer {layer}: a dynamic '{role}' needs both its static taps and its tap projection");
            }
            CheckShape(store, statics, layer, "dynamic conv static taps", conv.Kernel, channels);
            if (store.ShapeOf(tapProj.ObjectId, tapProj.TensorName) is not [var rows, _] || rows != (long)geometry.NumKvHeads * conv.Kernel)
            {
                throw new ContainerException($"layer {layer}: '{tapProj.TensorName}' is not [{geometry.NumKvHeads * conv.Kernel}, hidden]");
            }
            return new ResidualConvOp
            {
                StaticTaps = statics, TapProjection = tapProj, Channels = channels, Kernel = conv.Kernel, Heads = geometry.NumKvHeads,
            };
        }
        return BindStaticConv(store, stackId, layer, roles, role, conv.Kernel, channels);
    }

    private static ResidualConvOp? BindStaticConv(OperandStore store, string stackId, int layer, LayerRoleMap roles, string role,
        int kernel, int channels)
    {
        if (RoleOperand(store, stackId, layer, roles, role) is not { } weight)
        {
            return null;
        }
        var shape = store.ShapeOf(weight.ObjectId, weight.TensorName);
        if (!(shape.SequenceEqual(new long[] { channels, 1, kernel }) || shape.SequenceEqual(new long[] { channels, kernel })))
        {
            throw new ContainerException(
                $"layer {layer}: '{weight.TensorName}' [{string.Join(", ", shape)}] is not a depthwise conv of {channels} channels and width {kernel}");
        }
        return new ResidualConvOp { Weight = weight, Channels = channels, Kernel = kernel };
    }

    // ── packed experts ───────────────────────────────────────────────────────

    private static LayerFfn? BindPackedExperts(OperandStore store, string stackId, int layer, FfnSurface ffn, MoeSurface moe,
        LayerRoleMap roles, int hidden)
    {
        var router = RoleOperand(store, stackId, layer, roles, TensorRoles.Router);
        var gateUp = RoleOperand(store, stackId, layer, roles, TensorRoles.PackedGateUp);
        var down = RoleOperand(store, stackId, layer, roles, TensorRoles.PackedDown);
        if (router is null && gateUp is null && down is null)
        {
            return null;
        }
        if (router is null || gateUp is null || down is null)
        {
            throw new ContainerException(
                $"operand closure: layer {layer}: packed experts need a router and both projections (router {router is not null}, " +
                $"gate-up {gateUp is not null}, down {down is not null})");
        }
        if (moe.GateUpLayout is not GateUpLayouts.InterleavedRows)
        {
            throw new UnsupportedOperatorException($"layer {layer}: packed experts with gate-up layout '{moe.GateUpLayout}' have no managed implementation");
        }
        CheckShape(store, router, layer, "router", moe.Experts, hidden);
        CheckShape(store, gateUp, layer, "packed gate-up", moe.Experts, 2L * moe.ExpertIntermediateSize, hidden);
        CheckShape(store, down, layer, "packed down", moe.Experts, hidden, moe.ExpertIntermediateSize);
        var bias = RoleOperand(store, stackId, layer, roles, TensorRoles.Router, "bias");
        if (moe.SelectionBias && bias is null)
        {
            throw new ContainerException($"operand closure: layer {layer}: a selection bias is declared but the router has none");
        }
        if (moe.RoutingPolicy != ExpertRoutingPolicy.SigmoidNormalisedOverSelected)
        {
            throw new UnsupportedOperatorException($"layer {layer}: packed experts routed by {moe.RoutingPolicy} have no managed implementation");
        }
        PackedOperand Packed(string role, OperandRef elements) => new()
        {
            Elements = elements,
            Scale = RoleOperand(store, stackId, layer, roles, role, "scale"),
            Amax = RoleOperand(store, stackId, layer, roles, role, "amax"),
            Denominator = RoleOperand(store, stackId, layer, roles, role, "denominator"),
        };
        return new LayerFfn
        {
            Routed = new RoutedFfnOp
            {
                Router = router,
                NumExperts = moe.Experts,
                TopK = moe.TopK,
                ExpertIntermediateSize = moe.ExpertIntermediateSize,
                HiddenSize = hidden,
                Activation = ffn.Activation,
                RoutingPolicy = moe.RoutingPolicy,
                ExpertGatePrefix = string.Empty,
                ExpertUpPrefix = string.Empty,
                ExpertDownPrefix = string.Empty,
                Packed = new PackedExpertsOp
                {
                    GateUp = Packed(TensorRoles.PackedGateUp, gateUp),
                    Down = Packed(TensorRoles.PackedDown, down),
                    InterleavedGateUp = true,
                },
                SelectionBias = moe.SelectionBias ? bias : null,
                RouteScale = moe.RouteScale ?? 1.0,
                GatePolicy = ffn.GatePolicy,
            },
        };
    }

    private static void CheckShape(OperandStore store, OperandRef operand, int layer, string what, params long[] shape)
    {
        var actual = store.ShapeOf(operand.ObjectId, operand.TensorName);
        if (!actual.SequenceEqual(shape))
        {
            string at = layer >= 0 ? $"layer {layer}: " : string.Empty;
            throw new ContainerException(
                $"operand closure: {at}'{operand.ObjectId}/{operand.TensorName}' ({what}) has shape [{string.Join(", ", actual)}], " +
                $"expected [{string.Join(", ", shape)}]");
        }
    }
}
