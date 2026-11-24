using System.IO;
using System.Windows;
using System.Windows.Controls;
using Amql.Hf;
using Microsoft.Win32;

namespace Amql.Gui.Pages;

/// <summary>
/// Where a checkpoint comes from: a folder on this computer, or a Hugging Face
/// model (a URL or <c>owner/name</c>) that the CLI downloads before encoding.
/// The value is what goes on the command line either way. A token for gated or
/// private models is entered here, passed to the CLI as <c>HF_TOKEN</c>, and
/// never saved.
/// </summary>
public sealed class ModelSourceEditor : StackPanel
{
    private readonly RadioButton _local = new() { Content = "Folder on this computer", Margin = new Thickness(0, 0, 20, 0) };
    private readonly RadioButton _hub = new() { Content = "Hugging Face" };
    private readonly TextBox _folder = new();
    private readonly TextBox _url = new();
    private readonly PasswordBox _token = new() { MinWidth = 260, HorizontalAlignment = HorizontalAlignment.Left };
    private readonly StackPanel _localPanel = new();
    private readonly StackPanel _hubPanel = new();
    private readonly TextBlock _hubStatus = new();

    public event Action? ValueChanged;

    public ModelSourceEditor(string initial, Style captionStyle, Style iconStyle)
    {
        var choice = new StackPanel { Orientation = Orientation.Horizontal, Margin = new Thickness(0, 4, 0, 6) };
        string group = "source" + Guid.NewGuid().ToString("N");
        _local.GroupName = _hub.GroupName = group;
        choice.Children.Add(_local);
        choice.Children.Add(_hub);
        Children.Add(choice);

        // Local folder
        var row = new DockPanel();
        var browse = new Button
        {
            Content = new TextBlock { Text = "", Style = iconStyle },
            Margin = new Thickness(6, 0, 0, 0),
            ToolTip = "Choose the checkpoint folder (it holds config.json and *.safetensors)",
        };
        browse.Click += (_, _) =>
        {
            var dialog = new OpenFolderDialog { Title = "Hugging Face checkpoint folder" };
            if (Directory.Exists(_folder.Text)) dialog.InitialDirectory = _folder.Text;
            if (dialog.ShowDialog(Window.GetWindow(this)) == true) _folder.Text = dialog.FolderName;
        };
        DockPanel.SetDock(browse, Dock.Right);
        row.Children.Add(browse);
        row.Children.Add(_folder);
        _localPanel.Children.Add(row);
        _localPanel.Children.Add(new TextBlock
        {
            Text = "A checkpoint folder: config.json, *.safetensors and the tokenizer files.",
            Style = captionStyle,
            Margin = new Thickness(0, 4, 0, 0),
        });
        Children.Add(_localPanel);

        // Hugging Face
        _url.ToolTip = "https://huggingface.co/Qwen/Qwen3.5-0.8B, or just Qwen/Qwen3.5-0.8B (add @revision for a branch or commit)";
        _hubPanel.Children.Add(_url);
        _hubStatus.Style = captionStyle;
        _hubStatus.Margin = new Thickness(0, 4, 0, 8);
        _hubPanel.Children.Add(_hubStatus);
        var tokenRow = new StackPanel { Orientation = Orientation.Horizontal };
        tokenRow.Children.Add(new TextBlock { Text = "Access token", VerticalAlignment = VerticalAlignment.Center, Margin = new Thickness(0, 0, 10, 0) });
        tokenRow.Children.Add(_token);
        _hubPanel.Children.Add(tokenRow);
        _hubPanel.Children.Add(new TextBlock
        {
            Text = HfDownloader.ResolveToken() is null
                ? "Only needed for gated or private models. It is passed to amql-cli for this run and not saved."
                : "Only needed for gated or private models. A saved Hugging Face login was found and is used when this is empty.",
            Style = captionStyle,
            Margin = new Thickness(0, 4, 0, 0),
        });
        Children.Add(_hubPanel);

        bool isHub = initial.Length > 0 && !Directory.Exists(initial) && HfDownloader.TryParse(initial) is not null;
        (isHub ? _url : _folder).Text = initial;
        _local.IsChecked = !isHub;
        _hub.IsChecked = isHub;
        _local.Checked += (_, _) => Changed();
        _hub.Checked += (_, _) => Changed();
        _folder.TextChanged += (_, _) => Changed();
        _url.TextChanged += (_, _) => Changed();
        Refresh();
    }

    public bool IsHub => _hub.IsChecked == true;

    /// <summary>The argument for the command line.</summary>
    public string Value
    {
        get => (IsHub ? _url : _folder).Text.Trim();
        set
        {
            bool hub = value.Length > 0 && !Directory.Exists(value) && HfDownloader.TryParse(value) is not null;
            (hub ? _url : _folder).Text = value;
            _hub.IsChecked = hub;
            _local.IsChecked = !hub;
        }
    }

    /// <summary>The model the value names, when it is a Hugging Face one.</summary>
    public HfModelRef? Model => IsHub ? HfDownloader.TryParse(_url.Text.Trim()) : null;

    /// <summary>The token typed for this run, if any.</summary>
    public string? Token => IsHub && _token.Password.Trim() is { Length: > 0 } t ? t : null;

    private void Changed()
    {
        Refresh();
        ValueChanged?.Invoke();
    }

    private void Refresh()
    {
        _localPanel.Visibility = IsHub ? Visibility.Collapsed : Visibility.Visible;
        _hubPanel.Visibility = IsHub ? Visibility.Visible : Visibility.Collapsed;
        string text = _url.Text.Trim();
        _hubStatus.Text = text.Length == 0
            ? "Paste the model's page URL, or type owner/name."
            : HfDownloader.TryParse(text) is { } model
                ? $"Downloads {model} (safetensors, config and tokenizer files), then encodes it. It goes to {HfDownloader.DefaultDirectory(model)} unless a download folder is set under Advanced options. An interrupted download resumes."
                : "That is not a Hugging Face model URL or owner/name.";
    }
}
