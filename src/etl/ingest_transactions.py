
"""Ingest transaction CSV data and export it to Parquet.

The schema assumes the standard H&M transactions_train.csv layout.
Verify the real CSV header and data types before production ingestion.
"""

import argparse
import sys
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import (
    DateType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

# Resolve the repository root so direct script execution works from
# the project directory or another current working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.spark import get_spark_session


# Explicit provisional schema for the standard H&M transactions dataset.
TRANSACTIONS_SCHEMA = StructType(
    [
        StructField("t_dat", DateType(), nullable=True),
        StructField("customer_id", StringType(), nullable=True),
        StructField("article_id", StringType(), nullable=True),
        StructField("price", DoubleType(), nullable=True),
        StructField("sales_channel_id", IntegerType(), nullable=True),
    ]
)

REQUIRED_COLUMNS = [field.name for field in TRANSACTIONS_SCHEMA.fields]


def validate_input_file(input_path: str) -> Path:
    """Ensure the input exists and is a file."""
    path = Path(input_path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {path}")

    if not path.is_file():
        raise ValueError(f"Input path is not a file: {path}")

    if path.suffix.lower() != ".csv":
        raise ValueError(f"Expected a CSV input file, received: {path.name}")

    return path


def validate_csv_header(spark: SparkSession, input_path: Path) -> None:
    """Check that the CSV contains all expected columns."""
    header_df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "false")
        .csv(str(input_path))
    )

    actual_columns = header_df.columns
    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in actual_columns
    ]

    if missing_columns:
        raise ValueError(
            "CSV header does not match the expected transaction schema. "
            f"Missing columns: {missing_columns}. "
            f"Found columns: {actual_columns}. "
            "Verify the dataset before changing the schema."
        )


def read_transactions(
    spark: SparkSession,
    input_path: str,
) -> DataFrame:
    """Read transaction records using the explicit schema."""
    path = validate_input_file(input_path)
    validate_csv_header(spark, path)

    return (
        spark.read
        .option("header", "true")
        .option("mode", "FAILFAST")
        .option("dateFormat", "yyyy-MM-dd")
        .schema(TRANSACTIONS_SCHEMA)
        .csv(str(path))
    )


def ingest_transactions(input_path: str, output_path: str) -> int:
    """Ingest CSV transactions, export Parquet, and return the row count."""
    spark = get_spark_session(app_name="TransactionDataIngestion")

    try:
        transactions_df = read_transactions(spark, input_path)

        # Evaluate the input before writing so failures surface during ingestion.
        row_count = transactions_df.count()

        output = Path(output_path).expanduser().resolve()

        # Guard against accidentally targeting the input CSV itself.
        input_resolved = Path(input_path).expanduser().resolve()
        if output == input_resolved:
            raise ValueError("Output path must differ from the input CSV path.")

        output.parent.mkdir(parents=True, exist_ok=True)

        # Spark writes Parquet as a directory containing part files.
        transactions_df.write.mode("overwrite").parquet(str(output))

        print(f"Rows processed: {row_count:,}")
        print(f"Parquet output: {output}")
        transactions_df.printSchema()

        return row_count
    finally:
        spark.stop()


def main() -> None:
    """Parse command-line arguments and run ingestion."""
    parser = argparse.ArgumentParser(
        description="Ingest transaction CSV data and export it to Parquet."
    )
    parser.add_argument(
        "--input",
        default=str(PROJECT_ROOT / "data" / "raw" / "transactions_train.csv"),
        help="Path to the source CSV.",
    )
    parser.add_argument(
        "--output",
        default=str(
            PROJECT_ROOT / "data" / "processed" / "transactions.parquet"
        ),
        help="Destination Parquet directory.",
    )

    args = parser.parse_args()
    ingest_transactions(args.input, args.output)


if __name__ == "__main__":
    main()
