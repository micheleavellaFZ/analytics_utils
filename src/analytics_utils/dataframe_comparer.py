"""
A utility for comparing and analyzing differences between pandas DataFrames.

This package provides tools for detailed comparison of two DataFrames, helping users
identify differences in row count and content. It generates a structured comparison
result that highlights discrepancies and provides a summary of the differences.

Key Features:
- Row count comparison between two DataFrames
- Identification of unique rows in each DataFrame
- Detailed difference report with sample rows
- Column consistency validation
- Human-readable summary formatting
"""

import pandas as pd
from dataclasses import dataclass
from dataclasses import dataclass


@dataclass
class ComparisonResult:
    len_left: int
    len_right: int
    rows_only_in_left: int
    rows_only_in_right: int
    diff_dataframe: pd.DataFrame

    def summary_message(self) -> str:
        len_message = (
            f"Rows in left:\t\t{self.len_left}\nRows in right:\t\t{self.len_right}"
        )
        if self.len_left != self.len_right:
            diff = abs(self.len_left - self.len_right)
            more = "left" if self.len_left > self.len_right else "right"
            len_message += f"\n==>{more} has {diff} more"

        diff_message = f"Rows only in left:\t{self.rows_only_in_left}\nRows only in right:\t{self.rows_only_in_right}"

        out = "-" * 40 + f"\n{len_message}\n\n{diff_message}" + "\n" + "-" * 40 + "\n"

        return (
            out
            + self.diff_dataframe.sample(min(5, len(self.diff_dataframe))).to_string(
                index=False
            )
            + "\n"
            + "-" * 40
        )


def compare_dataframe(
    df_left: pd.DataFrame, df_right: pd.DataFrame
) -> ComparisonResult:
    check_columns(df_left, df_right)

    columns = list(df_left.columns.values)
    df_left["dataset"] = "left"
    df_right["dataset"] = "right"
    df_u = pd.concat([df_left, df_right], ignore_index=True)

    duplicated_mask = df_u.duplicated(subset=columns, keep=False)
    df_diff = df_u[~duplicated_mask].copy()

    more_on_left = (df_diff["dataset"] == "left").sum()
    more_on_right = (df_diff["dataset"] == "right").sum()

    return ComparisonResult(
        len(df_left), len(df_right), more_on_left, more_on_right, df_diff
    )


def check_columns(df_left: pd.DataFrame, df_right: pd.DataFrame) -> None:
    cols_left = set(df_left.columns)
    cols_right = set(df_right.columns)

    if cols_left != cols_right:
        un_on_left = cols_left - cols_right
        un_on_right = cols_right - cols_left
        raise ValueError(
            f"\nThe two DFs should have the same columns:\nunique on left DF: {un_on_left}\nunique on right DF: {un_on_right}"
        )


if __name__ == "__main__":
    df1 = pd.DataFrame({"a": ["a", None, "c", "f"], "b": ["a", "b", "c", "g"]})
    df2 = pd.DataFrame({"a": ["a", None, "c"], "b": ["a", "b", "c"]})
    df1["d"] = 3
    df2["c"] = 2

    comparison = compare_dataframe(df1, df2)
    print(comparison.summary_message())
