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
    public void A_Result_That_Comes_Back_After_Another_Is_Stale_Too()
    {
        // A harness answers a re-read with a note, then the file again, then the note again.
        string file = """{"content":"using System;"}""", note = """{"unchanged":true,"same_as_call":"call_c73f79"}""";
        var messages = new List<ChatMessage> { new("user", "go"), Call("read", """{"path":"P.cs"}"""), Result(file), Call("read", """{"path":"P.cs"}"""), Result(note) };
        Assert.Empty(RepeatedCalls.Stale(messages));
        messages.Add(Call("read", """{"path":"P.cs"}"""));
        messages.Add(Result(file));
        Assert.Single(RepeatedCalls.Stale(messages));
    }

    [Fact]
    public void A_Turn_Begins_At_The_Users_Message_Not_At_Notes_After_It()
    {
        var messages = new List<ChatMessage>
        {
            new("system", "s"), new("assistant", "[Context cleared]"),
            new("user", "write it with create_file"), new("user", "<|rag|> tool:edit_file"),   // the harness's notes
            Call("read", "{}"), Result("x"),
            new("user", "[System] same action 3 times"),   // a note inside the turn
            Call("read", "{}"), Result("x"),
        };
        Assert.Equal(2, ServerApp.TurnStart(messages));
        Assert.Equal(1, ServerApp.TurnStart([new("system", "s"), new("user", "hi")]));
        Assert.Equal(-1, ServerApp.TurnStart([new("system", "s")]));
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
