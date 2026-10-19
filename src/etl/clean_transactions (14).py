"""
Transaction cleaning and cold-start user identification.
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def clean_transactions(df: DataFrame) -> DataFrame:
    """Remove rows containing null values."""
    return df.dropna()


def identify_cold_start_users(
    df: DataFrame,
    user_column: str,
    threshold: int = 5,
) -> DataFrame:
    """Return users with fewer than threshold transactions."""
    user_counts = df.groupBy(user_column).count()
    return user_counts.filter(F.col("count") < threshold)
