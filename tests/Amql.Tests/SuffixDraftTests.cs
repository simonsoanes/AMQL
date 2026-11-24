using Amql.Server;
using Xunit;

namespace Amql.Tests;

public class SuffixDraftTests
{
    [Fact]
    public void The_Tokens_After_An_Earlier_Occurrence_Are_Drafted()
    {
        // ... 1 2 3 4 [5 6 7] ... 1 2 3 4: the ending 1 2 3 4 was followed by 5 6 7.
        var context = new List<int> { 9, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4 };
        Assert.Equal(new[] { 5, 6, 7 }, SuffixDraft.Find(context, 4, 3));
        Assert.Equal(new[] { 5, 6, 7, 8, 1, 2, 3, 4 }, SuffixDraft.Find(context, 4, 20));
    }

    [Fact]
    public void A_Short_Or_New_Ending_Drafts_Nothing()
    {
        var context = new List<int> { 1, 2, 3, 9, 1, 2, 3 };
        Assert.Null(SuffixDraft.Find(context, 4, 5));
        Assert.Equal(new[] { 9 }, SuffixDraft.Find(context, 3, 1));
        Assert.Null(SuffixDraft.Find(new List<int> { 5, 6, 7, 8 }, 2, 5));
        Assert.Null(SuffixDraft.Find(context, 0, 5));
    }

    [Fact]
    public void The_Longest_Match_Wins_Over_A_Later_Shorter_One()
    {
        // "0 1 2" was followed by 7 at its long match (0 1 2 after 5), and by 8 at a later short one.
        var context = new List<int> { 5, 0, 1, 2, 7, 3, 1, 2, 8, 4, 5, 0, 1, 2 };
        Assert.Equal(new[] { 7 }, SuffixDraft.Find(context, 2, 1));
    }
}
