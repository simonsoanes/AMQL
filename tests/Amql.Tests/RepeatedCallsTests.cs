using System.Text.Json;
using Amql.Cli;
using Amql.Server;
using Xunit;

namespace Amql.Tests;

public class RepeatedCallsTests
{
    private static ChatMessage Call(string name, string args) =>
        new("assistant", "") { ToolCalls = [new ChatToolCall(name, JsonDocument.Parse(args).RootElement.Clone())] };

    private static ChatMessage Result(string text) => new("tool", text);

    [Fact]
    public void A_Call_With_The_Same_Result_Twice_Is_Stale()
    {
        var messages = new List<ChatMessage>
        {
            new("user", "run it"),
            Call("run", """{"command":"dotnet run","wait_ms":5000}"""),
            Result("""{"output":"bin (0 MB)","durationMs":1077}"""),
            Call("run", """{"wait_ms":5000,"command":"dotnet run"}"""),   // the same call, keys in another order
            Result("""{"output":"bin (0 MB)","durationMs":1159}"""),   // the same result but for its timing
        };
        var stale = RepeatedCalls.Stale(messages);
        Assert.Contains(RepeatedCalls.Key("run", """{"command":"dotnet run","wait_ms":5000}"""), stale);
        Assert.DoesNotContain(RepeatedCalls.Key("run", """{"command":"dotnet build"}"""), stale);
    }

    [Fact]
    public void Once_Or_With_A_New_Result_Is_Not()
    {
        var once = new List<ChatMessage> { new("user", "go"), Call("build", "{}"), Result("failed: CS1002") };
        Assert.Empty(RepeatedCalls.Stale(once));
        var changed = new List<ChatMessage>(once) { Call("build", "{}"), Result("succeeded") };
        Assert.Empty(RepeatedCalls.Stale(changed));
        // after a change the same result twice again makes it stale
        changed.Add(Call("build", "{}"));
        changed.Add(Result("succeeded"));
        Assert.Single(RepeatedCalls.Stale(changed));
    }

    [Fact]
    public void Parallel_Calls_Pair_With_Their_Results_In_Order()
    {
        var two = new ChatMessage("assistant", "")
        {
            ToolCalls = [new ChatToolCall("a", JsonDocument.Parse("{}").RootElement.Clone()), new ChatToolCall("b", JsonDocument.Parse("{}").RootElement.Clone())],
        };
        var messages = new List<ChatMessage> { new("user", "go"), two, Result("A"), Result("B"), two, Result("A"), Result("B2") };
        var stale = RepeatedCalls.Stale(messages);
        Assert.Contains(RepeatedCalls.Key("a", "{}"), stale);
        Assert.DoesNotContain(RepeatedCalls.Key("b", "{}"), stale);
    }
}
