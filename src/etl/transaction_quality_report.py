
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def generate_quality_report(df: DataFrame) -> dict:
    """Generate basic data quality metrics for transactions."""

    total_rows = df.count()

    null_counts = {
        column: df.filter(F.col(column).isNull()).count()
        for column in df.columns
    }

    duplicate_rows = total_rows - df.dropDuplicates().count()

    return {
        "total_rows": total_rows,
        "duplicate_rows": duplicate_rows,
        "null_counts": null_counts,
    }
