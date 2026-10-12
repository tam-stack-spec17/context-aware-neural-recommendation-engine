
"""
Transaction cleaning module.

This module:
- Removes rows containing null values.
- Prepares transaction data for downstream processing.
- Supports cold-start user handling.
"""

from pyspark.sql import DataFrame


def clean_transactions(df: DataFrame) -> DataFrame:
    """
    Remove rows containing null values.

    Parameters:
        df: Input transaction DataFrame.

    Returns:
        Cleaned transaction DataFrame.
    """
    cleaned_df = df.dropna()

    return cleaned_df
