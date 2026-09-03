namespace Amql.Inference;

/// <summary>Raised when an operation the container declares has no managed
/// implementation in this build. Fail-closed: the primitive is named, never
/// approximated — mirroring the reference's exhaustive match refusing to
/// serve an unjudged operator.</summary>
public sealed class UnsupportedOperatorException : NotSupportedException
{
    public UnsupportedOperatorException(string message) : base(message) { }
}