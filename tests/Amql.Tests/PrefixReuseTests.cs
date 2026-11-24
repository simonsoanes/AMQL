using Amql.Server;
using Xunit;

namespace Amql.Tests;

/// <summary>Where the batch scheduler saves a prompt's state for the next
/// request to pick up from.</summary>
public class PrefixReuseTests
{
    private const int Turn = 9;

    [Fact]
    public void The_Mark_Is_At_The_Second_Last_Turn()
    {
        // system | user | notes | reply's turn: the notes move on the next call.
        int[] prompt = [Turn, 1, 2, Turn, 3, 4, Turn, 5, Turn, 6];
        Assert.Equal(6, PromptMark.At(prompt, Turn));
    }

    [Fact]
    public void A_Prompt_With_One_Turn_Is_Marked_There()
    {
        Assert.Equal(3, PromptMark.At([1, 2, 3, Turn, 6], Turn));
    }

    [Fact]
    public void Without_A_Turn_Marker_The_Mark_Leaves_One_Token()
    {
        Assert.Equal(4, PromptMark.At([1, 2, 3, 4, 5], Turn));
        Assert.Equal(4, PromptMark.At([1, 2, 3, 4, 5], null));
        Assert.Equal(2, PromptMark.At([Turn, 1, 2], Turn));
    }
}
