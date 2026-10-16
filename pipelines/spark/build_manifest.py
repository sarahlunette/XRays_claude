"""Spark manifest builder for millions of rows (revision plan 6.6).

Pixels stay out of Spark: it joins *metadata* and labels, dedupes, assigns splits and writes
a Parquet manifest partitioned by split. The split is computed with the exact same Python
function as the single-machine path (via mapInPandas), so both paths agree row for row.

    spark-submit pipelines/spark/build_manifest.py s3://lake/ s3://lake/manifests/v13/
"""

from __future__ import annotations

import sys

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from xrayml.data.manifest import _bucket

EXTERNAL_TEST_SITES = ["chu_partner_2026"]


def main(lake: str, out: str, val_pct: int = 10, test_pct: int = 15, salt: str = "split-v1") -> None:
    spark = (
        SparkSession.builder.appName("xrayml-manifest")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )
    images = spark.read.parquet(f"{lake}/images/")  # image_id, patient_hash, site_id, modality, phash, ...
    labels = spark.read.parquet(f"{lake}/labels/")  # image_id, fractured, boxes, polygons, reader
    sites = spark.read.parquet(f"{lake}/sites/")  # small dimension table

    df = (
        images.filter(F.col("modality").isin("CR", "DX"))  # narrow: no shuffle
        .dropDuplicates(["phash"])  # wide: shuffle on phash
        .join(labels, "image_id", "left")
        .join(F.broadcast(sites), "site_id")  # broadcast: no shuffle of the big table
        .withColumnRenamed("patient_hash", "group_id")
    )

    def assign(batches):
        for pdf in batches:
            b = pdf.group_id.map(lambda g: _bucket(str(g), salt))
            pdf["split"] = "train"
            pdf.loc[b < val_pct + test_pct, "split"] = "val"
            pdf.loc[b < test_pct, "split"] = "test"
            pdf.loc[pdf.site_id.isin(EXTERNAL_TEST_SITES), "split"] = "external_test"
            yield pdf

    df = df.mapInPandas(assign, schema=df.schema.add("split", "string"))
    df.groupBy("split", "body_part").agg(
        F.count("*").alias("n"), F.avg("fractured").alias("prevalence")
    ).show()
    df.repartition("split").write.mode("overwrite").partitionBy("split").parquet(out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
