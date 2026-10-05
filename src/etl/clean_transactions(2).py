
"""
Transaction cleaning module.

This module:
- Removes rows containing null values.
- Identifies cold-start users with fewer than 5 transactions.
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def clean_transactions(df: DataFrame) -> DataFrame:
    """
    Remove rows containing null values.
    """
    cleaned_df = df.dropna()
    return cleaned_df


def identify_cold_start_users(
    df: DataFrame,
    user_column: str,
    threshold: int = 5
) -> DataFrame:
    """
    Identify users having fewer than 5 transactions.
    """

    user_counts = (
        df.groupBy(user_column)
        .count()
    )

    cold_start_users = user_counts.filter(
        F.col("count") < threshold
    )

    return cold_start_users
