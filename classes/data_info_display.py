# data_info_display.py
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.box import ROUNDED

class DataInfoDisplay:
    """
    Class to display dataset information including column details, memory usage, and dataset dimensions.
    Colors are in blue-purple theme.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Constructor for the class.

        Args:
            data: Dataset as a pandas DataFrame
        """
        self.data = data
        self.console = Console()

        # Blue-purple color theme
        self.color_title_header = "#5E7AC2"  # SlateBlue
        self.color_high_missing = "#8A2BE2"  # BlueViolet
        self.color_low_missing = "#7B68EE"   # MediumSlateBlue
        self.color_no_missing = "#81818A"    # Lavender

    def _get_row_color(self, missing_pct: float) -> str:
        """
        Determine row color based on missing values percentage.

        Args:
            missing_pct: Percentage of missing values

        Returns:
            Appropriate color for the row
        """
        if missing_pct > 5:
            return self.color_high_missing
        elif missing_pct > 0:
            return self.color_low_missing
        else:
            return self.color_no_missing

    def _get_memory_usage(self) -> str:
        """
        Calculate memory usage of the dataset.

        Returns:
            Formatted string of memory usage
        """
        memory_bytes = self.data.memory_usage(deep=True).sum()
        memory_mb = memory_bytes / 1024 ** 2
        return f"Memory: {memory_mb:.2f} MB"

    def display(self):
        """
        Display complete dataset information including column table and dimensions.
        Dimensions are printed as plain text outside the table.
        """
        # Build main table
        table = Table(
            title="Data Info",
            title_style=f"bold {self.color_title_header}",
            box=ROUNDED,
            header_style=f"bold {self.color_title_header}",
            caption=f"  {self._get_memory_usage()}"
        )

        # Add columns to the table
        table.add_column("Column", no_wrap=True)
        table.add_column("Non-Null", justify="center")
        table.add_column("Unique", justify="center")
        table.add_column("Dtype", justify="center")
        table.add_column("Sample", justify="left", max_width=20)

        # Populate table rows
        for col in self.data.columns:
            total = len(self.data[col])
            non_null = self.data[col].count()
            null_count = total - non_null
            missing_pct = (null_count / total) * 100 if total > 0 else 0
            unique = self.data[col].nunique()
            sample = str(self.data[col].dropna().iloc[0])[:20] if non_null > 0 else "NaN"
            dtype = str(self.data[col].dtype)

            # Determine row color
            row_color = self._get_row_color(missing_pct)

            # Add row to table
            table.add_row(
                f"[{row_color}]{col}[/{row_color}]",
                f"[{row_color}]{non_null:,}[/{row_color}]",
                f"[{row_color}]{unique:,}[/{row_color}]",
                f"[{row_color}]{dtype}[/{row_color}]",
                f"[{row_color}]{sample}[/{row_color}]"
            )

        # Display the main table
        self.console.print(table)

        # Display dataset dimensions as plain text (not in a table)
               # Display dataset dimensions
        rows, cols = self.data.shape
        
        # زیباتر کردن بخش ابعاد
        dim_table = Table(show_header=False, box=None, padding=(0, 2))
        dim_table.add_column("Label", style=f"bold {self.color_title_header}", width=12)
        dim_table.add_column("Value", style="bold white", justify="right")

        dim_table.add_row("➤ Rows", f"{rows:,}")
        dim_table.add_row("➤ Columns", f"{cols:,}")

        self.console.print(dim_table)