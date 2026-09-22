using System.Windows;
using Amql.Gui.Model;
using Microsoft.Win32;

namespace Amql.Gui.Dialogs;

public partial class DefaultsDialog : Window
{
    public DefaultsDialog(ProjectDefaults defaults)
    {
        InitializeComponent();
        ContainerBox.Text = defaults.ContainerDir;
        TokenizerBox.Text = defaults.TokenizerDir;
        PatchBox.Text = defaults.PatchFile;
        ComponentBox.Text = defaults.Component;
    }

    public void ApplyTo(ProjectDefaults defaults)
    {
        defaults.ContainerDir = ContainerBox.Text.Trim();
        defaults.TokenizerDir = TokenizerBox.Text.Trim();
        defaults.PatchFile = PatchBox.Text.Trim();
        defaults.Component = string.IsNullOrWhiteSpace(ComponentBox.Text) ? "target" : ComponentBox.Text.Trim();
    }

    private void OnBrowseContainer(object sender, RoutedEventArgs e) =>
        ContainerBox.Text = PickDirectory("Container directory") ?? ContainerBox.Text;

    private void OnBrowseTokenizer(object sender, RoutedEventArgs e) =>
        TokenizerBox.Text = PickDirectory("Checkpoint (tokenizer) directory") ?? TokenizerBox.Text;

    private void OnBrowsePatch(object sender, RoutedEventArgs e)
    {
        var dialog = new OpenFileDialog
        {
            Title = "Select a weight patch",
            Filter = "Safetensors patch (*.safetensors)|*.safetensors|All files (*.*)|*.*",
            CheckFileExists = false,
        };
        if (dialog.ShowDialog(this) == true) PatchBox.Text = dialog.FileName;
    }

    private string? PickDirectory(string title)
    {
        var dialog = new OpenFolderDialog { Title = title };
        return dialog.ShowDialog(this) == true ? dialog.FolderName : null;
    }

    private void OnOk(object sender, RoutedEventArgs e) => DialogResult = true;
}
