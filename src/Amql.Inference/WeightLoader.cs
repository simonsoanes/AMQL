using Amql.Safetensors;
using Amql.Vindex3;

namespace Amql.Inference;

/// <summary>
/// Resolves <see cref="OperandRef"/>s from the operand store once, widens
/// them to f32 (the executor's dtype widening, mirroring the reference's
/// <c>load_weight</c>) and caches the matrices for the session lifetime.
/// </summary>
public sealed class WeightLoader
{
    private readonly OperandStore _store;
    private readonly Dictionary<(string ObjectId, string Tensor), Tensor2D> _matrices = new();
    private readonly Dictionary<(string ObjectId, string Tensor), float[]> _vectors = new();

    public WeightLoader(OperandStore store)
    {
        _store = store;
    }

    /// <summary>Loads a 2-D weight as [rows, cols] (weight convention:
    /// rows are the output space). The resolved element count must match
    /// the requested shape exactly — a shape relabel over identical bytes
    /// is a defect, not a convenience.</summary>
    public Tensor2D Matrix(OperandRef operand, int rows, int cols)
    {
        var key = (operand.ObjectId, operand.TensorName);
        if (_matrices.TryGetValue(key, out var cached))
        {
            return cached;
        }

        var resolution = _store.Resolve(operand);
        if (resolution.Shape.Length == 0 || ElementCount(resolution.Shape) != (long)rows * cols)
        {
            throw new ContainerException(
                $"operand '{operand.ObjectId}/{operand.TensorName}' resolves to shape " +
                $"[{string.Join(",", resolution.Shape)}] — expected [{rows} x {cols}]");
        }
        if (!resolution.Dtype.IsWidenableToF32())
        {
            throw new ContainerException(
                $"operand '{operand.ObjectId}/{operand.TensorName}' dtype {resolution.Dtype.Label()} " +
                "has no f32 widening path");
        }

        var data = BitPattern.WidenToF32(resolution.Dtype, resolution.Payload);
        var matrix = new Tensor2D(data, rows, cols);
        _matrices[key] = matrix;
        return matrix;
    }

    /// <summary>Loads a 1-D weight (norm scales).</summary>
    public float[] Vector(OperandRef operand, int width)
    {
        var key = (operand.ObjectId, operand.TensorName);
        if (_vectors.TryGetValue(key, out var cached))
        {
            return cached;
        }

        var resolution = _store.Resolve(operand);
        if (ElementCount(resolution.Shape) != width)
        {
            throw new ContainerException(
                $"operand '{operand.ObjectId}/{operand.TensorName}' resolves to " +
                $"{ElementCount(resolution.Shape)} elements — expected {width}");
        }
        if (!resolution.Dtype.IsWidenableToF32())
        {
            throw new ContainerException(
                $"operand '{operand.ObjectId}/{operand.TensorName}' dtype {resolution.Dtype.Label()} " +
                "has no f32 widening path");
        }

        var vector = BitPattern.WidenToF32(resolution.Dtype, resolution.Payload);
        _vectors[key] = vector;
        return vector;
    }

    public int LoadedMatrixCount => _matrices.Count;
    public int LoadedVectorCount => _vectors.Count;

    private static long ElementCount(long[] shape)
    {
        long n = 1;
        foreach (var dim in shape)
        {
            n = checked(n * dim);
        }
        return n;
    }
}