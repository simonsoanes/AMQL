using System.IO;
using System.Windows;
using Microsoft.Win32;

namespace Amql.Gui.Dialogs;

public partial class NewProjectDialog : Window
{
    public string ProjectName { get; private set; } = "Untitled AMQL Project";
    public string ProjectPath { get; private set; } = string.Empty;
    public string InitialContainer { get; private set; } = string.Empty;
    public string InitialTokenizer { get; private set; } = string.Empty;

    public NewProjectDialog()
    {
        InitializeComponent();
        PathBox.Text = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments),
            "AMQL Projects",
            "My AMQL Project" + Model.ProjectModel.FileExtension);
    }

    private void OnBrowsePath(object sender, RoutedEventArgs e)
    {
        var dialog = new SaveFileDialog
        {
            Filter = Model.ProjectModel.FileFilter,
            FileName = string.IsNullOrWhiteSpace(NameBox.Text) ? "project" : NameBox.Text,
        };
        if (dialog.ShowDialog(this) == true)
        {
            PathBox.Text = dialog.FileName;
            NameBox.Text = Path.GetFileNameWithoutExtension(dialog.FileName);
        }
    }

    private void OnBrowseContainer(object sender, RoutedEventArgs e) => ContainerBox.Text = PickDirectory("Container directory") ?? ContainerBox.Text;

    private void OnBrowseTokenizer(object sender, RoutedEventArgs e) => TokenizerBox.Text = PickDirectory("Checkpoint (tokenizer) directory") ?? TokenizerBox.Text;

    private string? PickDirectory(string title)
    {
        var dialog = new OpenFolderDialog { Title = title };
        return dialog.ShowDialog(this) == true ? dialog.FolderName : null;
    }

    private void OnCreate(object sender, RoutedEventArgs e)
    {
        var name = NameBox.Text.Trim();
        var path = PathBox.Text.Trim();
        if (name.Length == 0)
        {
            MessageBox.Show(this, "Give the project a name.", "New AMQL Project", MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }
        if (path.Length == 0)
        {
            MessageBox.Show(this, "Choose where to save the .amqlproj file.", "New AMQL Project", MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }
        try
        {
            var dir = Path.GetDirectoryName(path);
            if (!string.IsNullOrEmpty(dir))
            {
                Directory.CreateDirectory(dir);
            }
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, $"Cannot create the project folder:\n{ex.Message}", "New AMQL Project",
                MessageBoxButton.OK, MessageBoxImage.Error);
            return;
        }

        ProjectName = name;
        ProjectPath = path;
        InitialContainer = ContainerBox.Text.Trim();
        InitialTokenizer = TokenizerBox.Text.Trim();
        DialogResult = true;
    }
}
