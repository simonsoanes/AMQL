using System.Data;
using System.Globalization;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using Amql.Safetensors;

namespace Amql.Gui.Explorer;

public partial class TensorGridDialog : Window
{
    private float[] _values = Array.Empty<float>();
    private long[] _shape = Array.Empty<long>();
    private DataTable? _table;

    public TensorGridDialog(string objectId, string tensorName, TensorData data)
    {
        InitializeComponent();
        Owner = Application.Current.MainWindow;

        TensorNameText.Text = $"{objectId} / {tensorName}";
        TensorShapeText.Text = $"[{string.Join(" × ", data.Shape)}]";
        TensorDtypeText.Text = data.Dtype.Label();
        _values = data.Values;
        _shape = data.Shape;

        BuildGrid();

        // Selection tracking
        TensorGrid.SelectedCellsChanged += (_, _) => UpdateCellInfo();
    }

    private void BuildGrid()
    {
        _table = new DataTable();
        TensorGrid.Columns.Clear();

        if (_shape.Length == 1)
        {
            // Single column vector
            _table.Columns.Add("Index", typeof(int));
            _table.Columns.Add("Value", typeof(float));
            for (int i = 0; i < _shape[0]; i++)
            {
                _table.Rows.Add(i, _values[i]);
            }

            TensorGrid.Columns.Add(new DataGridTextColumn
            {
                Header = "#", Binding = new Binding("Index"), IsReadOnly = true, Width = 50,
            });
            TensorGrid.Columns.Add(new DataGridTextColumn
            {
                Header = "Value",
                Binding = new Binding("Value") { StringFormat = "F6" },
                Width = new DataGridLength(1, DataGridLengthUnitType.Star),
            });
        }
        else if (_shape.Length == 2)
        {
            // Matrix: rows × cols
            int rows = (int)_shape[0];
            int cols = (int)_shape[1];

            _table.Columns.Add("Row", typeof(int));
            for (int c = 0; c < cols; c++)
            {
                _table.Columns.Add($"Col {c}", typeof(float));
            }

            for (int r = 0; r < rows; r++)
            {
                var row = _table.NewRow();
                row[0] = r;
                for (int c = 0; c < cols; c++)
                {
                    row[c + 1] = _values[r * cols + c];
                }
                _table.Rows.Add(row);
            }

            TensorGrid.Columns.Add(new DataGridTextColumn
            {
                Header = "Row", Binding = new Binding("Row"), IsReadOnly = true, Width = 50,
            });

            // For large matrices, limit columns to avoid UI freeze
            int maxCols = Math.Min(cols, 256);
            for (int c = 0; c < maxCols; c++)
            {
                TensorGrid.Columns.Add(new DataGridTextColumn
                {
                    Header = $"C{c}",
                    Binding = new Binding($"Col {c}") { StringFormat = "F4" },
                    Width = 70,
                });
            }
            if (cols > maxCols)
            {
                StatsText.Text = $"Showing {maxCols} of {cols} columns. {rows} rows.";
            }
        }
        else
        {
            // Higher-rank: show as flat list with index
            _table.Columns.Add("Index", typeof(int));
            _table.Columns.Add("Value", typeof(float));
            long total = 1;
            foreach (var d in _shape) total *= d;
            for (int i = 0; i < total; i++)
            {
                _table.Rows.Add(i, _values[i]);
            }

            TensorGrid.Columns.Add(new DataGridTextColumn
            {
                Header = "#", Binding = new Binding("Index"), IsReadOnly = true, Width = 80,
            });
            TensorGrid.Columns.Add(new DataGridTextColumn
            {
                Header = "Value",
                Binding = new Binding("Value") { StringFormat = "F6" },
                Width = new DataGridLength(1, DataGridLengthUnitType.Star),
            });
        }

        TensorGrid.ItemsSource = _table.DefaultView;
        StatsText.Text = $"{_table.Rows.Count} entries";
    }

    private void UpdateCellInfo()
    {
        if (TensorGrid.SelectedCells.Count == 0)
        {
            CellInfoText.Text = "";
            return;
        }
        var cell = TensorGrid.SelectedCells[0];
        var col = cell.Column;
        var item = cell.Item as DataRowView;
        if (item is null) return;

        string val = col.DisplayIndex == 0
            ? item.Row[0]?.ToString() ?? ""
            : item.Row[col.DisplayIndex]?.ToString() ?? "";

        CellInfoText.Text = $"R{item.Row[0]} C{col.DisplayIndex - 1}: {val}";
    }

    private void OnFilterChanged(object sender, TextChangedEventArgs e)
    {
        if (_table is null) return;
        var view = _table.DefaultView;
        string filter = FilterBox.Text.Trim();
        if (filter.Length == 0)
        {
            view.RowFilter = "";
        }
        else if (float.TryParse(filter, NumberStyles.Float, CultureInfo.InvariantCulture, out float threshold))
        {
            // Filter rows where any value column has |val| > threshold
            var filters = new List<string>();
            for (int c = 1; c < _table.Columns.Count; c++)
            {
                string colName = _table.Columns[c].ColumnName;
                filters.Add($"[{colName}] > {threshold} OR [{colName}] < {-threshold}");
            }
            view.RowFilter = string.Join(" OR ", filters);
        }
        StatsText.Text = $"{view.Count} of {_table.Rows.Count} entries";
    }

    private void OnCellEditEnding(object? sender, DataGridCellEditEndingEventArgs e)
    {
        // Commit the edit to the underlying data
        if (e.EditingElement is TextBox tb)
        {
            if (float.TryParse(tb.Text, NumberStyles.Float, CultureInfo.InvariantCulture, out float val))
            {
                // Find the index in _values and update
                var rowView = e.Row.Item as DataRowView;
                if (rowView is null) return;
                int rowIndex = (int)rowView.Row[0];
                int colIndex = e.Column.DisplayIndex - 1; // skip row header column

                if (_shape.Length == 1 && colIndex == 0)
                {
                    _values[rowIndex] = val;
                    rowView.Row[1] = val;
                }
                else if (_shape.Length == 2 && colIndex >= 0)
                {
                    _values[rowIndex * (int)_shape[1] + colIndex] = val;
                    rowView.Row[colIndex + 1] = val;
                }
                else if (_shape.Length > 2 && colIndex == 0)
                {
                    _values[rowIndex] = val;
                    rowView.Row[1] = val;
                }
            }
        }
    }

    private void OnCopyCell(object sender, RoutedEventArgs e)
    {
        if (TensorGrid.SelectedCells.Count == 0) return;
        var cell = TensorGrid.SelectedCells[0];
        var rowView = cell.Item as DataRowView;
        if (rowView is null) return;
        string val = rowView.Row[cell.Column.DisplayIndex]?.ToString() ?? "";
        Clipboard.SetText(val);
    }

    private void OnCopyRow(object sender, RoutedEventArgs e)
    {
        if (TensorGrid.SelectedCells.Count == 0) return;
        var cell = TensorGrid.SelectedCells[0];
        var rowView = cell.Item as DataRowView;
        if (rowView is null) return;
        var parts = new List<string>();
        for (int c = 0; c < rowView.Row.ItemArray.Length; c++)
        {
            parts.Add(rowView.Row[c]?.ToString() ?? "");
        }
        Clipboard.SetText(string.Join("\t", parts));
    }

    private void OnClose(object sender, RoutedEventArgs e) => Close();
}