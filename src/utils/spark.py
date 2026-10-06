from pyspark.sql import SparkSession


def get_spark_session(
    app_name: str = "RecommendationSystem",
    driver_memory: str = "2g",
    executor_memory: str = "2g",
) -> SparkSession:
    """
    Create and return a reusable PySpark session.

    Parameters:
        app_name: Name of the Spark application.
        driver_memory: Memory allocated to the Spark driver.
        executor_memory: Memory allocated to Spark executors.

    Returns:
        SparkSession: Configured PySpark session.
    """

    spark = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.driver.memory", driver_memory)
        .config("spark.executor.memory", executor_memory)
        .config(
            "spark.sql.execution.arrow.pyspark.enabled",
            "true"
        )
        .getOrCreate()
    )

    return spark
