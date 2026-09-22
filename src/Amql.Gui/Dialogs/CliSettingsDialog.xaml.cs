using System.Windows;
using System.Windows.Controls;
using Amql.Gui.Model;
using Microsoft.Win32;

namespace Amql.Gui.Dialogs;

public partial class CliSettingsDialog : Window
{
    public CliSettingsDialog(CliSettings settings)
    {
        InitializeComponent();
        ExeBox.Text = settings.CliExePath ?? string.Empty;
        RepoBox.Text = settings.RepoPath ?? string.Empty;
        WeightsBox.Text = settings.WeightsEnv;
        foreach (ComboBoxItem item in DeviceBox.Items)
        {
            if ((item.Content as string) == settings.Device)
            {
                DeviceBox.SelectedItem = item;
                break;
            }
        }
        DeviceBox.SelectedItem ??= DeviceBox.Items[0];
    }

    public void ApplyTo(CliSettings settings)
    {
        settings.CliExePath = string.IsNullOrWhiteSpace(ExeBox.Text) ? null : ExeBox.Text.Trim();
        settings.RepoPath = string.IsNullOrWhiteSpace(RepoBox.Text) ? null : RepoBox.Text.Trim();
        settings.Device = (DeviceBox.SelectedItem as ComboBoxItem)?.Content as string ?? "auto";
        settings.WeightsEnv = WeightsBox.Text.Trim();
    }

    private void OnBrowseExe(object sender, RoutedEventArgs e)
    {
        var dialog = new OpenFileDialog
        {
            Title = "Select amql-cli.exe",
            Filter = "Executables (*.exe)|*.exe|All files (*.*)|*.*",
            CheckFileExists = true,
        };
        if (dialog.ShowDialog(this) == true) ExeBox.Text = dialog.FileName;
    }

    private void OnBrowseRepo(object sender, RoutedEventArgs e)
    {
        var dialog = new OpenFolderDialog { Title = "AMQL repository root" };
        if (dialog.ShowDialog(this) == true) RepoBox.Text = dialog.FolderName;
    }

    private void OnOk(object sender, RoutedEventArgs e) => DialogResult = true;
}
