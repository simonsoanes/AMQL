namespace Amql.Gui.Services;

/// <summary>
/// Where the shared services hand work back to the user interface. Both front
/// ends install themselves here at start-up: the WPF window posts to its
/// Dispatcher, the text UI to Terminal.Gui's main loop. Services that raise
/// events from process or network threads (<see cref="ServerController"/>)
/// go through it, so their events always arrive on the UI thread whichever
/// front end is running.
/// </summary>
public static class UiThread
{
    private static Func<bool> _isCurrent = () => true;
    private static Action<Action> _post = a => a();

    /// <summary>Installs the front end's dispatcher.</summary>
    public static void Install(Func<bool> isCurrent, Action<Action> post)
    {
        _isCurrent = isCurrent;
        _post = post;
    }

    /// <summary>Runs <paramref name="action"/> now when on the UI thread, otherwise queues it there.</summary>
    public static void Run(Action action)
    {
        if (_isCurrent())
        {
            action();
        }
        else
        {
            _post(action);
        }
    }

    /// <summary>Always queues, even from the UI thread (to run after the current event).</summary>
    public static void Post(Action action) => _post(action);
}
