
"""
Transaction cleaning module.

Removes null rows and separates cold-start users.
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def clean_transactions(df: DataFrame) -> DataFrame:
    """Remove rows containing null values."""
    return df.dropna()


def separate_cold_start_users(
    df: DataFrame,
    user_column: str,
    threshold: int = 5
):
    """
    Separate users with fewer than threshold transactions
    from users with threshold or more transactions.

    Returns:
        cold_start_users: User IDs with fewer than threshold transactions.
        regular_users: User IDs with threshold or more transactions.
    """
    user_counts = df.groupBy(user_column).count()

    cold_start_users = user_counts.filter(
        F.col("count") < threshold
    )

    regular_users = user_counts.filter(
        F.col("count") >= threshold
    )

    return cold_start_users, regular_users
